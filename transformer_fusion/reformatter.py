import re


def reformat(in_file, out_file=None):
    with open(in_file, 'r') as f_in:
        content = f_in.read()
    items = re.compile('\n|\t').split(content)
    n_items = len(items)
    new_content = ['\t'.join(items[i:i+8]) +
                   '\n' for i in range(0, n_items-7, 8)]
    if out_file:
        with open(out_file, 'w') as f_out:
            f_out.writelines(new_content)
    return new_content


def join(file_list, out_file=None):
    if not out_file:
        out_file = 'annotation_train.txt'
    content = []
    for file in file_list:
        file_content = reformat(file)
        if not content:
            content = file_content
        else:
            content += file_content[1:]
    with open(out_file, 'w') as f_out:
        f_out.writelines(content)


def main():
    join(['1_1_annotation.txt', '1_2_annotation.txt', '1_3_annotation.txt', '1_4_annotation.txt',
          '2_1_annotation.txt', '2_2_annotation.txt', '2_3_annotation.txt', '2_4_annotation.txt',
          '2_5_annotation.txt', '2_6__annotation.txt', '3_1_annotation.txt', '3_2_annotation.txt',
          '3_3_annotation.txt', '4_1_annotation.txt', '4_2_annotation.txt', '4_3_annotation.txt', '4_4_annotation.txt'])


if __name__ == '__main__':
    main()
