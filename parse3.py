import re
with open('/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/theory.tex') as f:
    text = f.read()

blocks = re.split(r'\\begin\{equation\}', text)
for i, block in enumerate(blocks[25:35]):
    eq = block.split('\\end{equation}')[0]
    print(f"Eq {i+26}: {eq.strip()}")

