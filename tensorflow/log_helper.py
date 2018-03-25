#encoding: utf8
import logging

logs = set()

def init_log(name, level = logging.INFO):
    if (name, level) in logs: return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    format_str = '%(asctime)s-%(filename)s#%(lineno)d:%(message)s'
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

#init_log('global')

def main():
    init_log('test')
    logger = logging.getLogger('test')
    logger.info([1,2,3,4])
    logger.debug('test debug')
if __name__ == '__main__':
    main()
