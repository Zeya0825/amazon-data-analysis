"""统一运行入口；任务在独立进程中执行，避免全局变量互相影响。"""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
TASKS = json.loads((PACKAGE_ROOT / 'config/tasks.json').read_text(encoding='utf-8'))
GROUPS = {
    'prepare': [4, 5, 6, 7, 8, 9],
    'explore': list(range(10, 23)) + [24, 25],
    'topics': [27, 28, 29, 30, 37],
    'recommend': [31, 32, 33, 34, 35, 38, 39],
    'business': [40, 41, 42, 43],
}
DEPENDENCIES = ['pandas', 'numpy', 'matplotlib', 'wordcloud', 'textblob',
                'gensim', 'scipy', 'scikit-surprise']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'check', 'task', 'group'])
    parser.add_argument('target', nargs='?')
    parser.add_argument('--root', type=Path, help='数据和运行输出根目录，例如D:\\Amazon_Project')
    parser.add_argument('--user', help='仅任务36：用户编号')
    parser.add_argument('--top', type=int, help='仅任务36：推荐数量')
    args = parser.parse_args()
    root = (args.root or PACKAGE_ROOT).expanduser().resolve()
    if args.command == 'list':
        for key, task in TASKS.items():
            print(f'{int(key):02d}  {task["name"]}')
        print('\n任务组：', GROUPS)
        return
    if args.command == 'check':
        print('Python:', sys.version.split()[0])
        print('解释器:', sys.executable)
        print('数据和运行输出目录:', root)
        for name in DEPENDENCIES:
            try:
                print(name, version(name))
            except PackageNotFoundError:
                print(name, '未安装')
        for rel in ['data_raw/sales/amazon.csv', 'data_raw/reviews/ratings_Electronics (1).csv',
                    'data_clean/products_unique.csv', 'data_clean/ratings_unique.csv']:
            print(rel, '存在' if (root / rel).is_file() else '不存在')
        print('check只检查环境和路径，不表示完整流程已经通过。')
        return
    if args.command == 'task':
        try:
            selected = [str(int(args.target))]
        except (TypeError, ValueError):
            parser.error('task后需要任务编号。')
        if selected[0] not in TASKS:
            parser.error('该编号未保留，请运行list查看。')
    else:
        if args.target not in GROUPS:
            parser.error('group后需要：' + ', '.join(GROUPS))
        selected = [str(n) for n in GROUPS[args.target]]
    if (args.user is not None or args.top is not None) and selected != ['36']:
        parser.error('--user和--top仅适用于task 36。')
    if args.top is not None and args.top < 1:
        parser.error('--top应为正整数。')
    env = os.environ.copy()
    env['AMAZON_PROJECT_ROOT'] = str(root)
    env['PYTHONPATH'] = str(PACKAGE_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUTF8'] = '1'
    for key in selected:
        print(f'开始任务{key}: {TASKS[key]["name"]}', flush=True)
        cmd = [sys.executable, '-m', TASKS[key]['module']]
        if args.user is not None:
            cmd += ['--user', args.user]
        if args.top is not None:
            cmd += ['--top', str(args.top)]
        subprocess.run(cmd, cwd=PACKAGE_ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
