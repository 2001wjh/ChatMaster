#!/bin/bash

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)/../..

# 运行单元测试
echo "运行单元测试..."
pytest test_retrieval_service.py -v

# 运行集成测试
echo "运行集成测试..."
pytest test_integration.py -v

# 运行所有测试并生成覆盖率报告
echo "运行所有测试并生成覆盖率报告..."
pytest --cov=../service --cov-report=html . 