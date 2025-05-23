//===- ExecutionPacketUtils.cpp - Execution Packet Utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for execution packet handling.
//
//===----------------------------------------------------------------------===//

#include "mtas/Dialect/Mt/Transforms/ExecutionPacketUtils.h"
#include <iomanip>

using namespace mlir;
using namespace mt;

bool ExecutionPacket::isFree() {
  return SMAC[0] == nullptr && SMAC[1] == nullptr && 
         SIEU == nullptr && SLDST == nullptr && SBR == nullptr && 
         VMAC[0] == nullptr && VMAC[1] == nullptr && VMAC[2] == nullptr &&
         VIEU == nullptr &&
         VLDST[0] == nullptr && VLDST[1] == nullptr;
}

bool ExecutionPacket::addOperation(mt::FunctionalUnit unit, InstructionSchedulingInterface op) {
  switch (unit) {
    case mt::FunctionalUnit::SMAC:
      if (SMAC[0] == nullptr) {
        SMAC[0] = op;
        return true;
      } else if (SMAC[1] == nullptr) {
        SMAC[1] = op;
        return true;
      } else {
        return false;
      }
      break;
    case mt::FunctionalUnit::SIEU:
      if(SIEU == nullptr) {
        SIEU = op;
        return true;
      } else {
        return false;
      }
      break;
    case mt::FunctionalUnit::SLDST:
      if(SLDST == nullptr) {
        SLDST = op;
        return true;
      } else {
        return false;
      }
      break;
    case mt::FunctionalUnit::SBR:
      if(SBR == nullptr) {
        SBR = op;
        return true;
      } else {
        return false;
      }
      break;
    case mt::FunctionalUnit::VMAC:
      if (VMAC[0] == nullptr) {
        VMAC[0] = op;
        return true;
      } else if (VMAC[1] == nullptr) {
        VMAC[1] = op;
        return true;
      } else if (VMAC[2] == nullptr) {
        VMAC[2] = op;
        return true;
      } else {
        return false;
      }
      break;
    case mt::FunctionalUnit::VIEU:
      if (VIEU == nullptr) {
        VIEU = op;
        return true;
      } else {
        return false;
      }
      break;
    case mt::FunctionalUnit::VLDST:
      if (VLDST[0] == nullptr) {
        VLDST[0] = op;
        return true;
      } else if (VLDST[1] == nullptr) {
        VLDST[1] = op;
        return true;
      } else {
        return false;
      }
      break;
    default:
      llvm_unreachable("Unknown functional unit");
  }
}

bool ExecutionPacket::addOperation(InstructionSchedulingInterface op) {
  auto functionalUnits = op.getFunctionalUnits();
  for (auto unit : functionalUnits) {
    if (addOperation(unit, op)) {
      return true; // 成功添加操作
    }
  }
  return false; // 没有可用的功能单元
}

ExecutionPacket ExecutionPacket::createFilteredCopy(const std::function<bool(InstructionSchedulingInterface)>& filterCondition) const {
  ExecutionPacket filteredPacket;
  
  // 过滤所有单元
  for (int i = 0; i < 2; ++i) {
    if (SMAC[i] && filterCondition(SMAC[i])) {
      filteredPacket.SMAC[i] = SMAC[i];
    }
  }
  
  if (SIEU && filterCondition(SIEU)) {
    filteredPacket.SIEU = SIEU;
  }
  
  if (SLDST && filterCondition(SLDST)) {
    filteredPacket.SLDST = SLDST;
  }
  
  if (SBR && filterCondition(SBR)) {
    filteredPacket.SBR = SBR;
  }
  
  for (int i = 0; i < 3; ++i) {
    if (VMAC[i] && filterCondition(VMAC[i])) {
      filteredPacket.VMAC[i] = VMAC[i];
    }
  }
  
  if (VIEU && filterCondition(VIEU)) {
    filteredPacket.VIEU = VIEU;
  }
  
  for (int i = 0; i < 2; ++i) {
    if (VLDST[i] && filterCondition(VLDST[i])) {
      filteredPacket.VLDST[i] = VLDST[i];
    }
  }
  
  return filteredPacket;
}

void ExecutionPacket::print(raw_ostream &os) {
  bool hasOperation = false;
  std::string firstPrefix = "\"    ";
  std::string followPrefix = "\"|   ";
  std::string suffix = "\\t\\n\"\n";
  
  // 检查并打印所有SMAC操作
  for (int i = 0; i < 2; ++i) {
    if (SMAC[i]) {
      std::string unitId = "M" + std::to_string(i+1); // SMAC编号: M1, M2
      if (!hasOperation) {
        os << firstPrefix << SMAC[i].formatOperationOutput(unitId) << suffix;
        hasOperation = true;
      } else {
        os << followPrefix << SMAC[i].formatOperationOutput(unitId) << suffix;
      }
    }
  }
  
  // 检查并打印SIEU操作
  if (SIEU) {
    if (!hasOperation) {
      os << firstPrefix << SIEU.formatOperationOutput("") << suffix;
      hasOperation = true;
    } else {
      os << followPrefix << SIEU.formatOperationOutput("") << suffix;
    }
  }
  
  // 检查并打印SLDST操作
  if (SLDST) {
    if (!hasOperation) {
      os << firstPrefix << SLDST.formatOperationOutput("") << suffix;
      hasOperation = true;
    } else {
      os << followPrefix << SLDST.formatOperationOutput("") << suffix;
    }
  }
  
  // 检查并打印SBR操作
  if (SBR) {
    if (!hasOperation) {
      os << firstPrefix << SBR.formatOperationOutput("") << suffix;
      hasOperation = true;
    } else {
      os << followPrefix << SBR.formatOperationOutput("") << suffix;
    }
  }
  
  // 检查并打印所有VMAC操作
  for (int i = 0; i < 3; ++i) {
    if (VMAC[i]) {
      std::string unitId = "M" + std::to_string(i+1); // VMAC编号: M1, M2, M3
      if (!hasOperation) {
        os << firstPrefix << VMAC[i].formatOperationOutput(unitId) << suffix;
        hasOperation = true;
      } else {
        os << followPrefix << VMAC[i].formatOperationOutput(unitId) << suffix;
      }
    }
  }
  
  // 检查并打印VIEU操作
  if (VIEU) {
    if (!hasOperation) {
      os << firstPrefix << VIEU.formatOperationOutput("") << suffix;
      hasOperation = true;
    } else {
      os << followPrefix << VIEU.formatOperationOutput("") << suffix;
    }
  }
  
  // 检查并打印所有VLDST操作
  for (int i = 0; i < 2; ++i) {
    if (VLDST[i]) {
      if (!hasOperation) {
        os << firstPrefix << VLDST[i].formatOperationOutput("") << suffix;
        hasOperation = true;
      } else {
        os << followPrefix << VLDST[i].formatOperationOutput("") << suffix;
      }
    }
  }
  
  // 如果执行包为空，输出SNOP
  if (!hasOperation) {
    std::ostringstream oss;
    oss << std::left << std::setw(18) << "SNOP";
    oss << std::left << std::setw(28) << "1";
    std::string snopOp = oss.str();
    os << firstPrefix << snopOp << suffix;
  }
}

int ExecutionPacket::calculateInstructionLength(){
    int totalLength = 0;
    
    // 遍历所有功能单元，累加指令长度
    // 标量单元
    for(int i = 0; i < 2; i++) {
        if(SMAC[i]) {
            totalLength += SMAC[i].getOperationSize();
        }
    }

    if(SIEU){
      totalLength += isa<mt::SmoviOp>(SIEU) ? 5 : SIEU.getOperationSize();
    }
    if(SLDST) totalLength += SLDST.getOperationSize();
    if(SBR) totalLength += SBR.getOperationSize();
    
    // 向量单元
    for(int i = 0; i < 3; i++) {
        if(VMAC[i]) {
            totalLength += VMAC[i].getOperationSize();
        }
    }
    if(VIEU) totalLength += VIEU.getOperationSize();
    for(int i = 0; i < 2; i++) {
        if(VLDST[i]) {
            totalLength += VLDST[i].getOperationSize();
        }
    }
    
    return totalLength ? totalLength : 5;
}

std::string mlir::mt::getSimplifiedOpName(InstructionSchedulingInterface op) {
  if (!op) return "-";
  
  Operation* operation = op.getOperation();
  std::string fullName = operation->getName().getStringRef().str();
  
  // 从完整名称中提取简化的操作类型
  int dotPos = fullName.find('.');
  if (dotPos != std::string::npos) {
    return fullName.substr(dotPos + 1);
  }
  return fullName;
}

std::string mlir::mt::padLeft(const std::string &str, unsigned width) {
  if (str.length() >= width) {
    return str.substr(0, width);
  }
  return str + std::string(width - str.length(), ' ');
}

void mlir::mt::printSchedulingWindowAsTable(
    const std::vector<ExecutionPacket>& window, 
    bool reverseOrder,
    int printStartPos,
    raw_ostream &os) {
  
  // 表格头部
  os << "+-----+---------------+-------+-------+-----+----------------------+-------+---------------+\n";
  os << "|Cycle|     SMAC      | SIEU  | SLDST | SBR |         VMAC         | VIEU  |     VLDST     |\n";
  os << "+-----+---------------+-------+-------+-----+----------------------+-------+---------------+\n";
  
  // 计算行迭代的范围
  int start = reverseOrder ? window.size() - 1 : 0;
  int end = reverseOrder ? -1 : window.size();
  int step = reverseOrder ? -1 : 1;
  
  // 表格内容
  for (int i = start; i != end; i += step) {
    const auto& packet = window[i];
    
    // 周期编号 - 根据是否逆序显示不同的值
    std::string cycleStr;
    if (reverseOrder) {
      cycleStr = std::to_string(printStartPos + int(window.size() - 1 - i));
    } else {
      cycleStr = std::to_string(printStartPos + int(i));
    }
    os << "| " << padLeft(cycleStr, 3) << " | ";
    
    // SMAC (2个)
    os << padLeft(getSimplifiedOpName(packet.SMAC[0]), 6) << " ";
    os << padLeft(getSimplifiedOpName(packet.SMAC[1]), 6) << " | ";
    
    // SIEU (1个)
    os << padLeft(getSimplifiedOpName(packet.SIEU), 5) << " | ";
    
    // SLDST (1个)
    os << padLeft(getSimplifiedOpName(packet.SLDST), 5) << " | ";
    
    // SBR (1个)
    os << padLeft(getSimplifiedOpName(packet.SBR), 3) << " | ";
    
    // VMAC (3个)
    os << padLeft(getSimplifiedOpName(packet.VMAC[0]), 6) << " ";
    os << padLeft(getSimplifiedOpName(packet.VMAC[1]), 6) << " ";
    os << padLeft(getSimplifiedOpName(packet.VMAC[2]), 6) << " | ";
    
    // VIEU (1个)
    os << padLeft(getSimplifiedOpName(packet.VIEU), 5) << " | ";
    
    // VLDST (2个)
    os << padLeft(getSimplifiedOpName(packet.VLDST[0]), 6) << " ";
    os << padLeft(getSimplifiedOpName(packet.VLDST[1]), 6) << " |\n";
  }
  
  // 表格底部
  os << "+-----+---------------+-------+-------+-----+----------------------+-------+---------------+\n";
}

void mlir::mt::printExecutionPackets(
    const std::vector<ExecutionPacket>& preLoopOps, 
    const std::vector<ExecutionPacket>& loopOps,
    const std::vector<ExecutionPacket>& postLoopOps, 
    raw_ostream &os){

    // 遍历preLoopOps
    for(int i = 0; i < preLoopOps.size(); i++){
      auto packet = preLoopOps[i];
      os << "// [" <<  i << "]" << "\n";
      // 打印packet
      packet.print(os);
    }
    // 输出循环标签
    os << "\"loop_k: \\t\\n\"\n";
    // 遍历loopOps
    for(int i = 0; i < loopOps.size(); i++){
      auto packet = loopOps[i];
      os << "// [" <<  i << "]" << "\n";
      // 打印packet
      packet.print(os);
    }
    // 遍历postLoopOps
    for(int i = 0; i < postLoopOps.size(); i++){
      auto packet = postLoopOps[i];
      os << "// [" <<  i << "]" << "\n";
      // 打印packet
      packet.print(os);
    }
}