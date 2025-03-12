# 练习

构建你自己的 GPT-4 分词器！

## 第一步

编写一个名为`BasicTokenizer`的类，包含以下三个核心函数：

- `def train(self, text, vocab_size, verbose=False)`：训练分词器。
- `def encode(self, text)`：将文本编码为分词后的 ID 序列。
- `def decode(self, ids)`：将 ID 序列解码回文本。

使用任意文本训练你的分词器，并可视化合并后的分词结果。这些分词是否看起来合理？你可以使用默认的测试文本文件`tests/taylorswift.txt`来进行测试。

## 第二步

将你的`BasicTokenizer`转换为一个`RegexTokenizer`，它使用正则表达式模式来精确地分割文本，就像 GPT-4 那样。分别处理各个部分，然后将结果拼接起来。重新训练你的分词器，并比较转换前后的结果。你将发现，现在不会再有跨越类别（数字、字母、标点符号、多个空格）的分词了。使用 GPT-4 的正则表达式模式：

```
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```


## 第三步

现在，你已经准备好加载 GPT-4 分词器的合并规则，并证明你的分词器在`encode`和`decode`上能够产生与[tiktoken]()完全相同的结果。


```python
# 示例代码
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # 这是 GPT-4 分词器
ids = enc.encode("hello world!!!? (안녕하세요!) lol123 😉")
text = enc.decode(ids)  # 解码后得到相同的文本
```

遗憾的是，你可能会遇到两个问题：

- 恢复原始合并规则并不容易：你可以轻松地恢复我们在这里称为`vocab`的内容，而他们在`enc._mergeable_ranks`中存储的内容也是如此。你可以直接复制`minbpe/gpt4.py`中的`recover_merges`函数，它会根据这些排名返回原始合并规则。如果你想了解该函数的工作原理，可以参考[这里]()和[这里]()。简而言之，在某些条件下，只需要存储父节点（以及它们的排名），而无需保留合并到任何父节点的子节点的详细信息。
- GPT-4 分词器对原始字节进行了重新排列：它将这种排列存储在`mergeable_ranks`的前 256 个元素中。因此，你可以相对简单地恢复这种字节排列：`byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}`。在你的`encode`和`decode`中，你需要相应地对字节进行重新排列。如果你遇到困难，可以参考`minbpe/gpt4.py`文件获取提示。

## 第四步

（可选，令人烦恼，且并非显然有用）添加处理特殊标记的能力。这样，你就可以匹配 tiktoken 的输出，即使存在特殊标记，例如：


```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # 这是 GPT-4 分词器
ids = enc.encode("<|endoftext|>hello world", allowed_special="all")
```


如果没有`allowed_special`参数，tiktoken 会报错。

## 第五步

如果你已经做到这一步，那么你已经成为 LLM 分词的专家了！遗憾的是，你还没有完全完成任务，因为除了 OpenAI 之外的许多 LLM（例如 Llama、Mistral）使用的是[sentencepiece]()而不是 BPE。主要区别在于，sentencepiece 直接在 Unicode 码点上运行 BPE，而不是在 UTF-8 编码的字节上。你可以自行探索 sentencepiece（祝你好运，它并不太美观）。如果你真的有时间，并且愿意接受挑战，可以尝试将你的 BPE 重写为基于 Unicode 码点的版本，并匹配 Llama 2 分词器。