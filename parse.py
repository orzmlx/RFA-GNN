import re
with open('/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/methodology.tex') as f:
    text = f.read()

# find blocks of \begin{equation} ... \end{equation} and adjacent text
blocks = re.split(r'\\begin\{equation\}', text)
for i, block in enumerate(blocks[1:]):
    eq = block.split('\\end{equation}')[0]
    print(f"Eq {i+1}: {eq.strip()}")

