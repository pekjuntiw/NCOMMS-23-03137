import json
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from spikingjelly.clock_driven import functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import raster_plot
from dataset import DeltaTransformedEEG, loader
from model import VO2LSNN


"""
ALL PARAMETERS SHOULD BE IN STANDARD S.I. UNITS TO PREVENT CONFUSION!
"""


def main():
    device = 'cuda'

    FLAGS = {
        "train_epoch": 150,
        "batch_size": 128,
        "lr": 1e-2,
        "decay_step": 130,
        "decay_rate": 0.8,
        "apply_class_weights": False,

        "num_in": 37,
        "num_lif": 24,
        "num_alif": 16,
        "num_out": 2,

        "save_model": True,
        "is_spike_reg": True,  # enable spike regularization

        "dt": 1.25e-3,
        "tau": 25e-3,
        "tau_lp": 25e-3,
        "max_delay": 10,  # syn delay in units of dt
        "refractory": 0,  # refractory in units of dt
        "out_cue_duration": 116,  # in units of dt
        "repetition": 1,

        # MOSFET parameters
        "kappa_n": 29e-6,
        "kappa_p": 18e-6,
        "vtn": 0.745,
        "vtp": 0.973,

        # VO2 LIF parameters
        "Vdd": 5.,
        "vth": 3.6,
        "vh": 1.5,
        "Rh": 14e3,
        "Rs": 1.5e3,
        "Cmem": 1.613e-6,
        "input_scaling": 5000e-6,

        # adaptation parameters
        "wl_ratio_n": 16,
        "wl_ratio_p": 8,
        "Ra": 100e3,
        "Ca": 12.5e-6,
    }

    save_model = FLAGS["save_model"]
    dataset_dir_train = 'data_eeg_balanced_2530'
    dataset_dir_test = 'data_eeg_contiguous'
    model_num = 1
    model_dir = f'eeg_vo2lsnn/{model_num}'
    model_output_dir = f'{model_dir}/model'
    log_dir = f'{model_dir}/log'
    fig_dir = f'{model_dir}/fig'
    model_output_name = f'{model_output_dir}/eeg_vo2lsnn'  # to be completed below

    # set if resume training from a saved model
    resume_training = False
    resume_from_epoch = 0

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # save flags
    if save_model:
        with open(f'{model_dir}/flags.json', 'w') as fp:
            json.dump(FLAGS, fp, sort_keys=True, indent=4)

    batch_size = FLAGS["batch_size"]
    test_batch_multiplier = 32
    train_batch_per_epoch = 20
    test_batch_per_epoch = 1
    train_epoch = FLAGS["train_epoch"]
    lr = FLAGS["lr"]
    decay_step = FLAGS["decay_step"]
    decay_rate = FLAGS["decay_rate"]
    apply_class_weights = FLAGS["apply_class_weights"]

    num_in = FLAGS["num_in"]
    num_lif = FLAGS["num_lif"]
    num_alif = FLAGS["num_alif"]
    num_out = FLAGS["num_out"]

    dt = FLAGS["dt"]
    tau = FLAGS["tau"]
    tau_lp = FLAGS["tau_lp"]
    max_delay = FLAGS["max_delay"]  # syn delay in units of dt
    refractory = FLAGS["refractory"]  # refractory in units of dt

    kappa_n = FLAGS["kappa_n"]
    kappa_p = FLAGS["kappa_p"]
    vtn = FLAGS["vtn"]
    vtp = FLAGS["vtp"]

    Vdd = FLAGS["Vdd"]
    vth = FLAGS["vth"]
    vh = FLAGS["vh"]
    Rh = FLAGS["Rh"]
    Rs = FLAGS["Rs"]
    Cmem = FLAGS["Cmem"]
    input_scaling = FLAGS["input_scaling"]

    wl_ratio_n = FLAGS["wl_ratio_n"]
    wl_ratio_p = FLAGS["wl_ratio_p"]
    Ra = FLAGS["Ra"]
    Ca = FLAGS["Ca"]

    out_cue_duration = FLAGS["out_cue_duration"]  # in units of dt
    repetition = FLAGS["repetition"]
    T = 1000 * repetition + out_cue_duration  # in units of dt

    is_spike_reg = FLAGS["is_spike_reg"]

    writer = SummaryWriter(log_dir)

    # initialize dataloader
    train_dataset = DeltaTransformedEEG(path=dataset_dir_train, output_cue_length=out_cue_duration)
    test_dataset = DeltaTransformedEEG(path=dataset_dir_test, output_cue_length=out_cue_duration)
    class_weights = torch.Tensor(compute_class_weight(
        'balanced' if apply_class_weights else None,
        classes=train_dataset.get_classes(),
        y=train_dataset.labels.numpy()
    )).to(device)

    train_data_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size * test_batch_multiplier, shuffle=False, drop_last=False,
    )

    # initialize VO2-based LSNN model
    net = VO2LSNN(num_in=num_in, num_lif=num_lif, num_alif=num_alif, num_out=num_out,
                  tau=tau, tau_lp=tau_lp,
                  Rh=Rh, Rs=Rs, Ra=Ra, Cmem=Cmem, Ca=Ca,
                  v_threshold=vth, v_reset=vh,
                  vtn=vtn, vtp=vtp, kappa_n=kappa_n, kappa_p=kappa_p,
                  wl_ratio_n=wl_ratio_n, wl_ratio_p=wl_ratio_p,
                  Vdd=Vdd,
                  input_scaling=input_scaling,
                  dt=dt, device=device, max_delay=max_delay, refractory=refractory)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

    # check size of model
    mem_params = sum([param.nelement() * param.element_size() for param in net.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in net.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    print(net)
    print(f"Params: {mem_params}bytes      Bufs: {mem_bufs}bytes      Total: {mem}bytes")
    if device == 'cuda':
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # load save model if resume training
    if resume_training:
        print(f'Loading model, scheduler and optimizer from {model_output_name}_ep{resume_from_epoch}.ckpt')
        chkpt = torch.load(f'{model_output_name}_ep{resume_from_epoch}.ckpt')
        net.load_state_dict(chkpt["net"])
        optimizer.load_state_dict(chkpt["optimizer"])
        scheduler.load_state_dict(chkpt["scheduler"])

    test_accs = np.load(f'{model_output_dir}/test_accs_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_accs = np.load(f'{model_output_dir}/train_accs_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_loss = np.load(f'{model_output_dir}/test_loss_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    train_loss = np.load(f'{model_output_dir}/train_loss_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_sensitivity = np.load(
        f'{model_output_dir}/test_sensitivity_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_specificity = np.load(
        f'{model_output_dir}/test_specificity_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_precision = np.load(
        f'{model_output_dir}/test_precision_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_g = np.load(f'{model_output_dir}/test_g_ep{resume_from_epoch}.npy').tolist() if resume_training else []
    test_f1 = np.load(f'{model_output_dir}/test_f1_ep{resume_from_epoch}.npy').tolist() if resume_training else []

    train_times = len(train_loss) if resume_training else 0
    test_times = len(test_loss) if resume_training else 0
    max_test_accuracy = max(test_accs) if resume_training else 0
    max_test_sensitivity = max(test_sensitivity) if resume_training else 0
    max_test_specificity = max(test_specificity) if resume_training else 0
    max_test_precision = max(test_precision) if resume_training else 0
    max_test_g = max(test_g) if resume_training else 0
    max_test_f1 = max(test_f1) if resume_training else 0
    print(train_times, test_times, max_test_accuracy, max_test_sensitivity, max_test_specificity, max_test_precision,
          max_test_g, max_test_f1)

    # store test confusion matrix
    confusion_matrix_test = np.zeros([num_out, num_out], dtype=int)

    start_epoch = resume_from_epoch + 1 if resume_training else 0
    for epoch in range(start_epoch, train_epoch):
        print(f"Epoch {epoch}: lr={scheduler.get_last_lr()}")
        train_correct_sum = 0
        train_sum = 0
        train_loss_sum = 0
        confusion_matrix_test[...] = 0

        # train model
        net.train()

        # get data in batches
        for eeg, label in (pbar := tqdm(loader(train_data_loader, device), total=train_batch_per_epoch)):
            optimizer.zero_grad()

            # forward pass
            output_full = net(eeg.float())
            output = output_full[-1]

            # compute loss, backward pass (BPTT)
            loss = F.cross_entropy(output, label, weight=class_weights)
            if is_spike_reg:
                spike_regularization = net.spike_regularization()
                loss += spike_regularization
            loss.backward()

            optimizer.step()
            scheduler.step()

            # reset SNN state after each forward pass
            functional.reset_net(net)

            # take output neuron with the largest activation in the last timestep as the classification result
            is_correct = (output.max(1)[1] == label).float()
            train_correct_sum += is_correct.sum().item()
            train_sum += label.numel()
            train_batch_accuracy = is_correct.mean().item()
            writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
            train_accs.append(train_batch_accuracy)

            train_loss_sum += loss.item()
            train_loss.append(loss.item())
            writer.add_scalar('train_batch_loss', loss, train_times)

            pbar.set_postfix_str(
                f'    (Step {train_times}: acc={train_batch_accuracy * 100:.4f}, loss={loss.item():.4f})'
            )

            train_times += 1

        train_accuracy = train_correct_sum / train_sum
        train_loss_avg = train_loss_sum / train_batch_per_epoch

        print("Testing...")

        # test model
        net.eval()

        with torch.no_grad():
            test_correct_sum = 0
            test_sum = 0
            test_loss_sum = 0

            # get data in batches
            for eeg, label in tqdm(loader(test_data_loader, device), total=test_batch_per_epoch):
                # forward pass
                output_full = net(eeg.float())
                output = output_full[-1]

                # compute loss
                loss = F.cross_entropy(output, label, weight=class_weights)
                if is_spike_reg:
                    spike_regularization = net.spike_regularization()
                    loss += spike_regularization

                # reset SNN state after each forward pass
                functional.reset_net(net)

                # take output neuron with the largest activation in the last timestep as the classification result
                test_correct_sum += (output.max(1)[1] == label).float().sum().item()
                test_sum += label.numel()

                test_loss_sum += loss.item()
                test_loss.append(loss.item())
                writer.add_scalar('test_batch_loss', loss, test_times)

                # calculate confusion matrix
                for target_label, predicted_label in zip(label.cpu().numpy(), output.max(1)[1].cpu().numpy()):
                    confusion_matrix_test[predicted_label, target_label] += 1

                test_times += 1

            test_accuracy = test_correct_sum / test_sum
            test_loss_avg = test_loss_sum / test_batch_per_epoch
            test_accs.append(test_accuracy)

            # calculate various metrics
            # TP: predicted +ve (1, epilepsy), actual +ve (1, epilepsy): confusion_matrix[1, 1]
            # TN: predicted -ve (0, normal  ), actual -ve (0, normal  ): confusion_matrix[0, 0]
            # FP: predicted +ve (1, epilepsy), actual -ve (0, normal  ): confusion_matrix[1, 0]
            # FN: predicted -ve (0, normal  ), actual +ve (1, epilepsy): confusion_matrix[0, 1]
            tp, tn = confusion_matrix_test[1, 1], confusion_matrix_test[0, 0]
            fp, fn = confusion_matrix_test[1, 0], confusion_matrix_test[0, 1]
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            g_mean = (sensitivity * specificity) ** 0.5
            f1_measure = 2 * tp / (2 * tp + fp + fn)
            test_sensitivity.append(sensitivity)
            test_specificity.append(specificity)
            test_precision.append(precision)
            test_g.append(g_mean)
            test_f1.append(f1_measure)

            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            writer.add_scalar('test_sensitivity', sensitivity, epoch)
            writer.add_scalar('test_specificity', specificity, epoch)
            writer.add_scalar('test_precision', precision, epoch)
            writer.add_scalar('test_g_mean', g_mean, epoch)
            writer.add_scalar('test_f1_measure', f1_measure, epoch)

            # save model if g-mean is improved
            if save_model:
                save_metrics = False
                if g_mean >= max_test_g:
                    print(
                        f'G-mean:     Saving net, scheduler state and optimizer state to {model_output_name}_ep{epoch}.ckpt')
                    chkpt = {
                        "net": net.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                    torch.save(chkpt, f'{model_output_name}_ep{epoch}.ckpt')
                    save_metrics = True

                if save_metrics:
                    print(f'Saving losses and accuracies to {model_output_dir}')
                    np.save(f'{model_output_dir}/train_accs_ep{epoch}.npy', np.array(train_accs))
                    np.save(f'{model_output_dir}/test_accs_ep{epoch}.npy', np.array(test_accs))
                    np.save(f'{model_output_dir}/train_loss_ep{epoch}.npy', np.array(train_loss))
                    np.save(f'{model_output_dir}/test_loss_ep{epoch}.npy', np.array(test_loss))
                    np.save(f'{model_output_dir}/test_sensitivity_ep{epoch}.npy', np.array(test_sensitivity))
                    np.save(f'{model_output_dir}/test_specificity_ep{epoch}.npy', np.array(test_specificity))
                    np.save(f'{model_output_dir}/test_precision_ep{epoch}.npy', np.array(test_precision))
                    np.save(f'{model_output_dir}/test_g_ep{epoch}.npy', np.array(test_g))
                    np.save(f'{model_output_dir}/test_f1_ep{epoch}.npy', np.array(test_f1))

            max_test_accuracy = max(max_test_accuracy, test_accuracy)
            max_test_sensitivity = max(max_test_sensitivity, sensitivity)
            max_test_specificity = max(max_test_specificity, specificity)
            max_test_precision = max(max_test_precision, precision)
            max_test_g = max(max_test_g, g_mean)
            max_test_f1 = max(max_test_f1, f1_measure)

            print(
                f"Epoch {epoch}: train_acc = {train_accuracy:.6f}, test_acc = {test_accuracy:.6f} (max {max_test_accuracy:.6f}), train_loss_avg = {train_loss_avg:.6f}, test_loss_avg = {test_loss_avg:.6f}, train_times = {train_times}\n"
                f"test_sensitivity = {sensitivity:.6f} (max {max_test_sensitivity:.6f}), test_specificity = {specificity:.6f} (max {max_test_specificity:.6f}), test_precision = {precision:.6f} (max {max_test_precision:.6f})\n"
                f"test_g_mean = {g_mean:.6f} (max {max_test_g:.6f}), test_f1_measure = {f1_measure:.6f} (max {max_test_f1:.6f})")
            print()

    # plot some figures
    net.eval()

    with torch.no_grad():
        wave = test_dataset.get_vr(0)
        eeg, label = test_dataset[0]
        eeg = torch.unsqueeze(eeg, dim=0)
        eeg = eeg.to(device)

        # forward pass
        output_full = net(eeg.float())

        # get model dynamical states
        output_evol = np.squeeze(output_full.cpu().numpy())
        v_evol = net.v.cpu().numpy()
        v_threshold_evol = net.v_threshold.cpu().numpy()
        s_evol = net.spike.cpu().numpy()

        plt.figure()
        raster_plot(np.arange(eeg.shape[1]), eeg.cpu()[0], show=False, xlim=(-10, eeg.shape[1] + 10))
        plt.figure()
        plt.plot(np.arange(wave.shape[0]), wave + np.tile(np.arange(wave.shape[1]), (wave.shape[0], 1)))
        plt.figure()
        plt.plot(np.arange(v_evol.shape[0]), v_evol[:, :min(10, num_lif)])
        plt.title("V LIF")
        plt.savefig(f'{fig_dir}/lif-vmem.png')
        plt.figure()
        plt.plot(np.arange(v_evol.shape[0]), v_evol[:, num_lif:(num_lif + min(10, num_alif))])
        plt.title("V ALIF")
        plt.savefig(f'{fig_dir}/alif-vmem.png')
        plt.figure()
        plt.plot(np.arange(v_threshold_evol.shape[0]), v_threshold_evol[:, :min(10, num_alif)])
        plt.title("Vth ALIF")
        plt.savefig(f'{fig_dir}/alif-va.png')
        plt.figure()
        plt.plot(np.arange(s_evol.shape[0]),
                 s_evol[:, :min(10, num_lif)] + 2 * np.tile(np.arange(min(10, num_lif)), (s_evol.shape[0], 1)))
        plt.title("Spike LIF")
        plt.savefig(f'{fig_dir}/lif-spike.png')
        plt.figure()
        plt.plot(np.arange(s_evol.shape[0]),
                 s_evol[:, num_lif:(num_lif + min(10, num_alif))] + 2 * np.tile(np.arange(min(10, num_alif)),
                                                                                (s_evol.shape[0], 1)))
        plt.title("Spike ALIF")
        plt.savefig(f'{fig_dir}/alif-spike.png')
        plt.figure()
        plt.plot(np.arange(output_evol.shape[0]), output_evol)
        plt.title("Out")
        plt.savefig(f'{fig_dir}/out.png')
        plt.figure()
        plt.plot(np.arange(output_evol.shape[0]), torch.softmax(torch.tensor(output_evol), dim=-1))
        plt.title("Softmax Out")
        plt.savefig(f'{fig_dir}/out-softmax.png')
        plt.figure()
        plt.imshow(torch.softmax(torch.tensor(output_evol), dim=-1).T, aspect='auto', interpolation='none')
        plt.colorbar()
        plt.savefig(f'{fig_dir}/out-softmax-heatmap.png')
        plt.figure()
        raster_plot(np.arange(s_evol.shape[0]), s_evol, show=False, xlim=(-10, s_evol.shape[0] + 10))
        plt.savefig(f'{fig_dir}/spike-raster.png')

    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    test_sensitivity = np.array(test_sensitivity)
    test_specificity = np.array(test_specificity)
    test_precision = np.array(test_precision)
    test_g = np.array(test_g)
    test_f1 = np.array(test_f1)
    if save_model:
        np.save(f'{model_dir}/train_accs.npy', train_accs)
        np.save(f'{model_dir}/test_accs.npy', test_accs)
        np.save(f'{model_dir}/train_loss.npy', train_loss)
        np.save(f'{model_dir}/test_loss.npy', test_loss)
        np.save(f'{model_dir}/test_sensitivity.npy', test_sensitivity)
        np.save(f'{model_dir}/test_specificity.npy', test_specificity)
        np.save(f'{model_dir}/test_precision.npy', test_precision)
        np.save(f'{model_dir}/test_g.npy', test_g)
        np.save(f'{model_dir}/test_f1.npy', test_f1)

    plt.figure()
    plt.plot(np.arange(train_accs.shape[0]), train_accs)
    plt.title("Train accuracy")
    plt.savefig(f'{fig_dir}/accs-train.png')
    plt.figure()
    plt.plot(np.arange(test_accs.shape[0]), test_accs)
    plt.title("Test accuracy")
    plt.savefig(f'{fig_dir}/accs-test.png')
    plt.figure()
    plt.plot(np.arange(train_loss.shape[0]), train_loss)
    plt.title("Train loss")
    plt.savefig(f'{fig_dir}/loss-train.png')
    plt.figure()
    plt.plot(np.arange(test_loss.shape[0]), test_loss)
    plt.title("Test loss")
    plt.savefig(f'{fig_dir}/loss-test.png')
    plt.figure()
    plt.plot(np.arange(test_sensitivity.shape[0]), test_sensitivity)
    plt.title("Test sensitivity")
    plt.savefig(f'{fig_dir}/sensitivity-test.png')
    plt.figure()
    plt.plot(np.arange(test_specificity.shape[0]), test_specificity)
    plt.title("Test specificity")
    plt.savefig(f'{fig_dir}/specificity-test.png')
    plt.figure()
    plt.plot(np.arange(test_precision.shape[0]), test_precision)
    plt.title("Test precision")
    plt.savefig(f'{fig_dir}/precision-test.png')
    plt.figure()
    plt.plot(np.arange(test_g.shape[0]), test_g)
    plt.title("Test g-mean")
    plt.savefig(f'{fig_dir}/g-test.png')
    plt.figure()
    plt.plot(np.arange(test_f1.shape[0]), test_f1)
    plt.title("Test f1-measure")
    plt.savefig(f'{fig_dir}/f1-test.png')
    plt.figure()
    sns.heatmap(
        confusion_matrix_test,
        annot=True, fmt="d", xticklabels=test_dataset.get_classes(False), yticklabels=test_dataset.get_classes(False)
    )
    plt.gca().invert_yaxis()
    plt.yticks(rotation=0)
    plt.savefig(f'{fig_dir}/confusion.png')
    # plt.show()

    print(f'Done! Check model at {model_dir}/')


if __name__ == '__main__':
    main()
