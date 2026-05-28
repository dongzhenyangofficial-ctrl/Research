from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy

import os
import time
import paddle
import numpy as np

def val_epoch(epoch, data_loader, model, criterion, opt, writer, optimizer):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, target, audio, visualization_item, bool_masked_pos, batch_size = process_data_item(opt, data_item)
        bool_masked_pos = None
        data_time.update(time.time() - end_time)
        with paddle.no_grad():
            output, loss = run_model(opt, [visual, target, audio, bool_masked_pos], model, criterion, i, print_attention=False)

        # F = features[:, 0]
        # T = paddle.to_tensor(np.load('Text_vector.npy'))
        # F /= F.norm(axis=-1, keepdim=True)
        # T /= T.norm(axis=-1, keepdim=True)
        # output = (100.0 * F @ T.t())

        acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc', accuracies.avg, epoch)
    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))

    # save_file_path = os.path.join(opt.ckpt_path, 'save_{}.pth'.format(epoch))
    # states = {
    #     'epoch': epoch + 1,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }
    # paddle.save(states, save_file_path)
