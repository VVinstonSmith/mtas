#include "mtas/Dialect/Mt/IR/Mt.h"
#include "mtas/Dialect/Ftm/IR/Ftm.h"
#include "mtas/Dialect/Mt/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
#define GEN_PASS_DEF_ALLOCATEADDRESSREGISTERS
#include "mtas/Dialect/Mt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mt;

namespace {

void implAllocateAddressRegisters(func::FuncOp funcOp) {
    auto ctx = funcOp.getContext();
    OpBuilder builder(ctx);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());

    // 1. 遍历函数的参数，判断其中的参数的ftm.memory_level属性
    for(auto arg : funcOp.getArguments()) {
        // 如果arg是llvm指针类型
        if(arg.getType().isa<LLVM::LLVMPointerType>()) {
            // 判断arg是否有ftm.memory_level属性
            if(!funcOp.getArgAttr(arg.getArgNumber(), ftm::MemLevelAttr::name)) {
                continue;
            }
            // 获取ftm.memory_level属性
            auto memLevelAttr = funcOp.getArgAttrOfType<ftm::MemLevelAttr>(
                arg.getArgNumber(), ftm::MemLevelAttr::name);
            auto memLevel = memLevelAttr.getLevel();
            if(memLevel != ftm::Cache::AM && memLevel != ftm::Cache::SM) {
                continue;
            }
            
            ftm::DeclareRegisterOp declareAR = 
                builder.create<ftm::DeclareRegisterOp>(funcOp.getLoc(), LLVM::LLVMPointerType::get(ctx));
            if(memLevel == ftm::Cache::SM) {
                declareAR->setAttr(ftm::MemLevelAttr::name, 
                    ftm::MemLevelAttr::get(ctx, ftm::Cache::ScalarAddressRegister));
            } else if(memLevel == ftm::Cache::AM) {
                declareAR->setAttr(ftm::MemLevelAttr::name, 
                    ftm::MemLevelAttr::get(ctx, ftm::Cache::VectorAddressRegister));
            }
            // 替换所有对该参数的使用为对ftm.address_register的使用
            arg.replaceAllUsesWith(declareAR);
            // 使用mt.SMVAGA将参数的值存储到分配的ftm.address_register中
            builder.create<mt::SmvagaOp>(funcOp.getLoc(), 
                arg, declareAR);
        }
    }
    
    // 遍历for循环
    funcOp.walk([&](scf::ForOp forOp) {
        builder.setInsertionPoint(forOp);
        // 记录所有的非!llvm.ptr类型的迭代参数
        SmallVector<Value> newInitArgs;
        for(auto initArg : forOp.getInitArgs()) {
            if(!initArg.getType().isa<LLVM::LLVMPointerType>()) {
                newInitArgs.push_back(initArg);
            }
        }
        // 如果没有!llvm.ptr类型的迭代参数，直接返回
        if(newInitArgs.size() == forOp.getInitArgs().size()) {
            return WalkResult::advance();
        }
        // 创建新的for循环，不再保留!llvm.ptr类型的迭代参数
        auto newForOp = builder.create<scf::ForOp>(
            forOp.getLoc(), 
            forOp.getLowerBound(), 
            forOp.getUpperBound(), 
            forOp.getStep(), 
            newInitArgs,
            [&](OpBuilder &nestedBuilder, Location loc, Value iv, ValueRange iterArgs) {
                // 创建映射
                IRMapping mapper;
                // 映射迭代变量
                mapper.map(forOp.getInductionVar(), iv);
                // 映射迭代参数
                // 逻辑是如果是!llvm.ptr类型，映射到对应的ftm.address_register
                // 否则映射到原来的迭代参数，但是这里需要注意的是，由于删除了!llvm.ptr类型的迭代参数，
                // 所以映射的位置关系不再是简单的索引位置一一对应
                size_t index = 0;
                for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); ++i) {
                    Value oldArg = forOp.getRegionIterArgs()[i];
                    Value oldArgInit = forOp.getInitArgs()[i];
                    if(!oldArg.getType().isa<LLVM::LLVMPointerType>()){
                        // 如果不是!llvm.ptr类型，直接映射到新的迭代参数
                        mapper.map(oldArg, iterArgs[index++]);
                        continue;
                    } else {
                        // 如果是!llvm.ptr类型，映射到对应到原迭代参数的初始值
                        mapper.map(oldArg, oldArgInit);
                    }
                }
                
                // 克隆循环体中的操作
                for (auto &op : forOp.getRegion().front()) {
                    // 如果是ftm::adda操作
                    if(auto addaOp = dyn_cast<ftm::AddaOp>(op)){
                        // 使用mt::adda（有副作用，无返回值）替换ftm::adda（无副作用，有返回值）
                        auto lhs = mapper.lookupOrDefault(addaOp.getLhs());
                        auto rhs = mapper.lookupOrDefault(addaOp.getRhs());
                        nestedBuilder.create<mt::SaddaOp>(op.getLoc(), 
                            rhs, lhs, lhs);
                        continue;
                    }

                    if(auto yieldOp = dyn_cast<scf::YieldOp>(op)){
                        // 如果是yield操作，删除对!llvm.ptr类型的迭代参数的使用
                        // 记录所有的非!llvm.ptr类型的迭代参数
                        SmallVector<Value> newYieldOperands;
                        for(auto operand : yieldOp.getOperands()){
                            if(!operand.getType().isa<LLVM::LLVMPointerType>()){
                                newYieldOperands.push_back(operand);
                            }
                        }
                        nestedBuilder.create<scf::YieldOp>(op.getLoc(), 
                            newYieldOperands);
                        continue;
                    }
                    
                    // 如果是其他操作，克隆操作
                    nestedBuilder.clone(op, mapper);
                }
            });

        // 移除旧循环
        forOp.erase();
    });
}

} // namespace

namespace mlir {
class AllocateAddressRegistersPass : public impl::AllocateAddressRegistersBase<AllocateAddressRegistersPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    implAllocateAddressRegisters(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mt::createAllocateAddressRegistersPass() {
  return std::make_unique<AllocateAddressRegistersPass>();
}