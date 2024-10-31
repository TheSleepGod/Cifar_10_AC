# Cifar_10_AC
BUAA 智能安全大作业


## 1 题目要求
* 平台不公开的实现有若干判别模型，且针对初始的500张样本，判别模型预测成功率均为100%。
* 提交一个攻击后数据集的zip压缩包，目录树应与下方相同
* 参赛者可以无需提交label.txt文件，即使提交也不会影响结果

### 1.1 数据集
* 本赛事从 Cifar-10 数据集中筛选了 500 张 32 * 32 尺寸的图像
* 图像命名方式为 X.png ，其中 X 为 [0, 500) 范围内的整型数字
* 标签文件 label.txt，其中每一行代表：<图像名称> <图像类别>
```
|-- images
	|-- 0.png
	|-- 1.png
	|-- …
	|-- 499.png
|-- label.txt
```
## 2 攻击思路
### 2.1 模型训练
* 使用cifar全集训练模型，利用模型进行梯度反向传播，以加入扰动进行攻击

## 附录
**1 Git规范**
```
# 拉取最新dev分支
git checkout dev
git pull origin

# 签出开发(或修复)分支
git checkout -b feature/xxx (or fix/xxx)

# 提交修改
git add .
git commit -m "[feat](xxx): message" (or "[fix](xxx): message")

# 解决与dev分支的冲突
git checkout dev
git pull origin
git checkout feature/xxx (or fix/xxx)
git rebase dev
git add .
git rebase --continue

# 提交开发/修复分支
git push origin
(pull request on github.com)

# 删除开发/修复分支
git checkout dev
git pull origin
git branch -d feature/xxx (or fix/xxx)
```
