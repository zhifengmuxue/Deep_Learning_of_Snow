import subprocess
from graphviz import render


def save_dot_graph(dot_code, file_path):
    # 保存 dot 代码到文本文件
    with open(file_path, 'w') as f:
        f.write(dot_code)

    # 使用 dot 命令执行生成图像
    image_file_path = file_path.rsplit('.', 1)[0] + '.png'
    return image_file_path


dot_code = '''digraph g {
    2601492327776 [label="() float64", color = orange, style = filled]
    2601492718592 [label="Add", color = lightblue, style = filled, shape = box]
    2601492781088 -> 2601492718592
    2601486942512 -> 2601492718592
    2601492718592 -> 2601492327776
    2601492781088 [label="() float64", color = orange, style = filled]
    2601486942512 [label="() float64", color = orange, style = filled]
}'''

file_path = 'graph.dot'
image_file_path = save_dot_graph(dot_code, file_path)
render('dot', 'png', 'graph.dot')
print(f'图像文件已保存为：{image_file_path}')
