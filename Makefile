# Makefile

# 默认目标
.PHONY: test clean

# 测试目标
test:
	PYTHONPATH=$(PWD)/src python -m unittest discover -s src/tests -p "*.py"

# 清理目标（删除所有的 pyc 文件）
clean:
	find . -name "*.pyc" -exec rm -f {} \;
	find . -name "__pycache__" -exec rm -rf {} \;
