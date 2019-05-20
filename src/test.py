import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            if i > 0:
                break
            data, target = data.to(device), target.to(device)
            output = model.get_features(data)
            for n, module in model.named_modules():
                module.weight = torch.nn.Parameter(module.weight.type(torch.uint8).to(device))
                if module.bias is not None:
                    module.bias = torch.nn.Parameter(module.bias.type(torch.uint8).to(device))
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
