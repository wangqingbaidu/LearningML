#Python

```python
newDir = filepath + '/' + file # 将文件命加入到当前文件路径后面
```
尽量不要使用这种字符串拼接的方式进行路径的生成，使用`os.path.join`会更安全。这种当时一旦`filepath=''`你这样拼接就会出现风险。
