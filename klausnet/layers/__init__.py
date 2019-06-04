"""
Layers
- Dense
- CNN
- RNN

### 目前，Dense Layer是自带一个activation的！！！！！
但我们也可以把activation 作为一个新的layer，直接放到model里。
"""


from klausnet.layers.fully_connected import Dense
from klausnet.layers.convolution import Conv2D, MaxPooling2D
from klausnet.layers.recurrent import SimpleRNNCell
from klausnet.layers.base import Layer