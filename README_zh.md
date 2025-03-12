# minbpe

这是一个极简、干净的字节级的 Byte Pair Encoding (BPE) 算法实现，BPE 算法常用于大语言模型 (LLM) 的分词器中。该算法是“字节级”的，因为它在 UTF-8 编码的字符串上运行。

BPE 算法是由 OpenAI 的[GPT-2 论文]()及其相关的[GPT-2 代码]()推广到大型语言模型领域的。[Sennrich 等人 2015 年]()的论文是 BPE 在自然语言处理（NLP）应用中的原始参考文献。如今，所有现代大型语言模型（例如 GPT、Llama、Mistral）都使用这种算法来训练它们的分词器。

这个仓库中有两种分词器，它们都可以执行分词器的 3 个主要功能：1）在给定文本上训练分词器的词汇表和合并规则；2）将文本编码为 Tokens；3）将 Tokens 解码为文本。仓库中的文件如下：

1. [minbpe/base.py](minbpe/base.py)：实现了`Tokenizer`基类，包含`train`、`encode`和`decode`的存根，以及保存/加载功能，还有一些通用的工具函数。这个类不建议直接使用，而是作为继承的基础。
2.  [minbpe/basic.py](minbpe/basic.py)：实现了`BasicTokenizer`，这是 BPE 算法的最简单实现，直接在文本上运行。
3. [minbpe/regex.py](minbpe/regex.py)：实现了`RegexTokenizer`，它通过正则表达式模式进一步分割输入文本，这是预处理阶段，会根据类别（例如：字母、数字、标点符号）分割输入文本，以确保合并不会跨越类别边界。这一方法最初在 GPT-2 论文中提出，并一直沿用到 GPT-4。该类还处理特殊标记（如果有的话）。
4. [minbpe/gpt4.py](minbpe/gpt4.py)：实现了`GPT4Tokenizer`。这是一个轻量级的`RegexTokenizer`包装器（见第 2 点），能够完全复现[tiktoken]()库中 GPT-4 的分词结果。包装器处理了一些细节，例如恢复分词器中的确切合并规则，以及处理一些不幸的（可能是历史遗留的？）1 字节标记排列。

最后，脚本[train.py](train.py)在输入文本[tests/taylorswift.txt](tests/taylorswift.txt)（这是她的维基百科词条，哈哈）上训练两种主要分词器，并将词汇表保存到磁盘以便可视化。在我的（M1）MacBook 上，这个脚本运行大约需要 25 秒。

以上所有文件都非常简短，并且有详细的注释，文件底部还包含使用示例。

## 快速开始

以最简单的例子为例，我们可以复现[维基百科上关于 BPE 的文章]()如下：


```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3) # 256 是字节标记的数量，然后进行 3 次合并
print(tokenizer.encode(text))
# [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
tokenizer.save("toy")
# 写入两个文件：toy.model（用于加载）和 toy.vocab（用于查看）
```

根据维基百科，对输入字符串"aaabdaaabac"运行 BPE，进行 3 次合并，结果是字符串"XdXac"，其中 X=ZY，Y=ab，Z=aa。需要注意的是，minbpe 总是将 256 个单独的字节分配为标记，然后根据需要合并字节。因此，对于我们来说，a=97，b=98，c=99，d=100（它们的[ASCII]()值）。然后，当(a,a)合并为 Z 时，Z 将成为 256。同样，Y 将成为 257，X 将成为 258。因此，我们从 256 个字节开始，进行 3 次合并，得到上述结果，预期输出为[258, 100, 258, 97, 99]。

## 推理：与 GPT-4 的比较

我们可以验证`RegexTokenizer`是否与[tiktoken]()中的 GPT-4 分词器具有功能一致性，如下所示：

```python
text = "hello123!!!? (안녕하세요!) 😉"

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

# ours
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]
```

（你需要运行`pip install tiktoken`来安装 tiktoken）。在底层，`GPT4Tokenizer`只是一个轻量级的`RegexTokenizer`包装器，传递 GPT-4 的合并规则和特殊标记。我们还可以确保正确处理特殊标记：

```python
text = "<|endoftext|>hello world"

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text, allowed_special="all"))
# [100257, 15339, 1917]

# ours
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text, allowed_special="all"))
# [100257, 15339, 1917]
```

需要注意的是，就像 tiktoken 一样，我们必须明确声明使用和解析特殊标记的意图。否则，这可能会成为一个重大隐患，意外地将攻击者控制的数据（例如用户提示）用特殊标记进行分词。`allowed_special`参数可以设置为"all"、"none"或允许的特殊标记列表。

## 训练

与 tiktoken 不同，这个代码允许你训练自己的分词器。原则上，据我所知，如果你在大型数据集上训练`RegexTokenizer`，并设置词汇表大小为 10 万，你将能够复现 GPT-4 分词器。

你可以选择两条路径。首先，如果你不想处理正则表达式模式分割和预处理文本的复杂性，也不关心特殊标记，那么可以选择`BasicTokenizer`。你可以训练它，然后进行编码和解码，例如：


```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
tokenizer.train(very_long_training_string, vocab_size=4096)
tokenizer.encode("hello world") # 字符串 -> tokens
tokenizer.decode([1000, 2000, 3000]) # tokens -> 字符串
tokenizer.save("mymodel") # 写入 mymodel.model 和 mymodel.vocab
tokenizer.load("mymodel.model") # 从磁盘加载模型，vocab 仅用于可视化
```


如果你希望遵循 OpenAI 在他们的文本分词器中采用的方法，即使用正则表达式模式按类别分割文本，那么`RegexTokenizer`是一个不错的选择。GPT-4 的模式是`RegexTokenizer`的默认设置，因此你可以简单地这样做：


```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)  # 使用一个很长的训练字符串进行训练，词汇表大小为 32768
tokenizer.encode("hello world")  # 将字符串编码为标记
tokenizer.decode([1000, 2000, 3000])  # 将标记解码为字符串
tokenizer.save("tok32k")  # 保存模型和词汇表，生成 tok32k.model 和 tok32k.vocab 文件
tokenizer.load("tok32k.model")  # 从磁盘加载模型
```

当然，你可以根据数据集的大小调整词汇表的大小。

**特殊标记**。最后，你可能希望为分词器添加特殊标记。可以通过`register_special_tokens`函数进行注册。例如，如果你的词汇表大小为 32768，那么前 256 个标记是原始字节标记，接下来的 32768-256 个是合并标记，之后可以添加特殊标记。最后一个“真实”的合并标记的 ID 为 32767（vocab_size-1），因此你的第一个特殊标记应该紧随其后，ID 为 32768。例如：


```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.register_special_tokens({"<|endoftext|>": 32768})  # 注册特殊标记
tokenizer.encode("<|endoftext|>hello world", allowed_special="all")  # 编码时允许使用特殊标记
```

当然，你也可以根据需要添加更多特殊标记。最后，我想强调，我努力保持代码的清晰、可读和易于修改。你不必害怕阅读代码并理解其工作原理。测试代码也是查看更多使用示例的好地方。

## 测试

我们使用 pytest 库进行测试。所有测试文件都位于`tests/`目录中。如果你还没有安装 pytest，可以使用以下命令安装：


```bash
$ pip install pytest
```


然后运行以下命令执行测试：


```bash
$ pytest -v .
```


（`-v`表示详细模式，输出更美观）

## 社区扩展


• [gnp/minbpe-rs]()：`minbpe`的 Rust 实现，与 Python 版本几乎完全对应。

## 练习

对于那些想要学习 BPE 的人，这里有一个逐步练习的建议，帮助你逐步构建自己的 minbpe。请参考[exercise.md](exercise.md)文件。