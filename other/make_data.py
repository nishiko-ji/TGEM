import os
from tqdm import tqdm
from log import Log


def make_data_train(data_dir, data_name, log_dir, log_files):
    with open(f'{data_dir}{data_name}_train.txt', 'w', encoding='UTF-8') as f:
        with tqdm(log_files) as pbar:
            for i, file in enumerate(pbar):
                pbar.set_description(f"[make_data_train({data_name})]")
                log = Log(log_dir+file)
                log.read_log()
                log.init_pieces_list()
                log.to_pieces_list()
                moves = ','.join(log.moves)
                label = []
                for l in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                    if l in log.red_pos0:
                        label.append('r')
                    else:
                        label.append('b')
                label2 = []
                en_colors = log.get_colors_last()
                for l in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                    label2.append(en_colors[l])
                l = ','.join(label)
                l2 = ','.join(label2)

                f.write(f'{l} {l2} {moves}\n')


def make_data_eval(data_dir, data_name, log_dir, log_files):
    with open(f'{data_dir}{data_name}_eval.txt', 'w', encoding='UTF-8') as f:
        with tqdm(log_files) as pbar:
            for i, file in enumerate(pbar):
                pbar.set_description(f"[make_data_eval({data_name})]")
                log = Log(log_dir+file)
                log.read_log()
                log.init_pieces_list()
                log.to_pieces_list()
                moves = ','.join(log.moves)
                label = []
                for l in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                    if l in log.red_pos0:
                        label.append('r')
                    else:
                        label.append('b')
                label2 = []
                for l in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                    if l in log.red_pos1:
                        label2.append('r')
                    else:
                        label2.append('b')
                l = ','.join(label)
                l2 = ','.join(label2)

                f.write(f'{l} {l2} {moves}\n')


def main():
    data_names = [
            'Naotti_hayazashi',
            'hayazashi_Naotti',
            'Naotti_Naotti',
            ]
    for data_name in data_names:
        log_dir = f'../log/{data_name}/log/'
        data_dir = '../data/'
        log_files = os.listdir(log_dir)

        make_data_train(data_dir, data_name, log_dir, log_files)
        make_data_eval(data_dir, data_name, log_dir, log_files)


if __name__ == '__main__':
    main()
