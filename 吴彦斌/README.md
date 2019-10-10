#Python

[hw1-2](hw1-2.py)

```python
if show_hidden == True:
    for fn in os.listdir(spec_directory):
        num = num + 1
```
这种统计直接使用`len`函数最好，所有具有`__iter__`方法的都可以使用`len`函数。


#### Tips： 不要上次rar的文件，直接是py文件就行。