# rl-bigwatermelon

目前使用A3C算法，离散动作域（后期会更换算法并改为连续域）

合成大西瓜游戏来自 https://github.com/bullhe4d/bigwatermelon，修改`ads.js`去除广告，防止agent学习到错误的state

目前`env.py`中识别游戏结束的方式还有待修改，目前方式可能会学习到错误的state