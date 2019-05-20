import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    # build model architecture
    config['arch']['args']['n_class'] = config['data_loader']['args']['n_class']
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = torch.nn.MSELoss()

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    model.load_backbone(checkpoint)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            if i > 0:
                break
            data, target = data.to(device), target.to(device)
            model.train()
            output = model.get_features(data)

            model.eval()
            # TODO: implement Integer-Arithmetic-Only Inference
            qout = model.get_features(data)

            loss = loss_fn(output, qout)
            logger.info(f"Loss: {loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-c', '--configs', default=None, type=str,
                        nargs='+',
                        help='config file paths (default: None)')

    config = ConfigParser(parser)
    main(config)
