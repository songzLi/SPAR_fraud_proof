import numpy as np
import matplotlib.pyplot as plt
KB = 1024
MB = KB ** 2


def cost(block_size, data_symbol_size, batch_factor, hash_size, code_rate,
         spar_header_size,
         stopping_ratio_2drs, stopping_ratio_spar_strong,
         stopping_ratio_spar_weak, target_prob, parity_size):
        # general parameters
        K = int(block_size / data_symbol_size)
        N = int(K / code_rate)
        B_data = data_symbol_size
        R = code_rate

        # 2D-RS specific parameters
        sqrt_N = int(np.ceil(np.sqrt(N)))
        num_layers_2drs = int(np.ceil(np.log2(sqrt_N)))
        merkel_proof_size_2drs = num_layers_2drs * hash_size
        # print('2drs num layers:', num_layers_2drs)
        # print('2drs proof size:', merkel_proof_size_2drs)

        # SPAR specific parameters
        C = batch_factor
        rf = C * R  # laye reduction factor
        # B_hash = hash_size * Cx
        num_layers_spar = int(np.ceil(np.log2(N / spar_header_size) /
                                      np.log2(rf)))
        merkel_proof_size_spar = hash_size * (C - 1) * num_layers_spar

        # 2D-RS costs
        header_2drs = 2 * sqrt_N * hash_size
        num_samples_2drs = num_samples(target_prob, stopping_ratio_2drs, N)
        sampling_bytes_2drs = \
            num_samples_2drs * (B_data + merkel_proof_size_2drs)
        incorrect_coding_proof_size_2drs = \
            int(np.ceil(np.sqrt(K))) * (B_data + merkel_proof_size_2drs)

        # SPAR costs
        header_spar = hash_size * spar_header_size
        print('header spar:', hash_size, spar_header_size, header_spar)
        num_samples_spar_strong = num_samples(target_prob,
                                              stopping_ratio_spar_strong, N)
        sampling_bytes_spar_strong = \
            num_samples_spar_strong * (B_data +
                                       merkel_proof_size_spar * (2 - R))
        num_samples_spar_weak = num_samples(target_prob,
                                            stopping_ratio_spar_weak, N)
        print(num_samples_2drs, num_samples_spar_strong, num_samples_spar_weak)
        sampling_bytes_spar_weak = \
            num_samples_spar_weak * (B_data +
                                     merkel_proof_size_spar * (2 - R))
        incorrect_coding_proof_size_spar = \
            parity_size * (B_data + merkel_proof_size_spar)
        print('block size:', int(block_size / MB), 'k:', K)
        print('2DRS:', int(sampling_bytes_2drs / KB))
        print('SPAR:', int(sampling_bytes_spar_strong / KB))
        return [header_2drs, sampling_bytes_2drs,
                incorrect_coding_proof_size_2drs,
                header_spar, sampling_bytes_spar_strong,
                sampling_bytes_spar_weak,
                incorrect_coding_proof_size_spar]


def block_size_vs_cost(block_size_range, data_symbol_size_range,
                       batch_factor, hash_size, code_rate,
                       spar_header_size, stopping_ratio_2drs,
                       stopping_ratio_spar_strong, stopping_ratio_spar_weak,
                       target_prob, parity_size):
    keywords = ['header_2drs', 'sampling_bytes_2drs',
                'incorrect_coding_proof_2drs',
                'header_spar', 'sampling_bytes_spar_strong',
                'sampling_bytes_spar_weak',
                'incorrect_coding_proof_spar']
    results = {}
    for k in keywords:
        results[k] = []

    for b, d in zip(block_size_range, data_symbol_size_range):
        costs = cost(b, d, batch_factor, hash_size, code_rate,
                     spar_header_size,
                     stopping_ratio_2drs, stopping_ratio_spar_strong,
                     stopping_ratio_spar_weak, target_prob,
                     parity_size)
        for k, c in zip(keywords, costs):
            results[k].append(c)

    for k in keywords:
        results[k] = np.array(results[k])

    block_size_range = np.array(block_size_range) / MB
    titles = ['header', 'sampling_bytes', 'incorrect_coding_proof']
    # plot absolute costs
    x_vectors = [block_size_range] * 3
    x_label = 'block size (MB)'
    for title in titles:
        if title == 'sampling_bytes':
            schemes = ['2drs', 'spar_strong', 'spar_weak']
            colors = ['orange', 'c', 'g']
        else:
            schemes = ['2drs', 'spar']
            colors = ['orange', 'g']

        # plot abs
        print(schemes)
        y_vectors = [results[title + '_' + s] / KB for s in schemes]
        labels = schemes + [None]
        y_label = title + ' (kB)'
        myPlot(x_vectors, y_vectors, colors, labels, x_label, y_label,
               title + ' size', scale='log')

        # plot costs normalized to block size
        y_vectors = [results[title + '_' + s] / block_size_range / MB
                     for s in schemes]
        labels = schemes
        title = title + ' to block size ratio'
        y_label = title
        myPlot(x_vectors, y_vectors, colors, labels, x_label, y_label,
               title + ' size', scale='log')

    # x_vector = block_size_range
    # y_vector_2drs = results['header_2drs'] + results['sampling_bytes_2drs']
    # y_vector_spar = results['header_spar'] + results['sampling_bytes_spar']
    # plt.semilogy(x_vector, y_vector_2drs / KB, label='2D-RS', marker='o')
    # plt.semilogy(x_vector, y_vector_spar / KB, label='SPAR', marker='s')
    # plt.xlabel('block size (MB)')
    # plt.ylabel('header + sampling (KB)')
    # plt.grid()
    # plt.legend(loc='best')
    # plt.title('header + sampling cost comparison')
    # plt.show()

    # plot incorrect coding cost
    # x_vector = block_size_range
    # y_vector_2drs = results['incorrect_coding_proof_2drs']
    # y_vector_spar = results['incorrect_coding_proof_spar']
    # plt.semilogy(x_vector, y_vector_2drs / KB, label='2D-RS', marker='o')
    # plt.semilogy(x_vector, y_vector_spar / KB, label='SPAR', marker='s')
    # plt.xlabel('block size (MB)')
    # plt.ylabel('proof size (KB)')
    # plt.grid()
    # plt.legend(loc='best')
    # plt.title('incorrect-coding proof size comparison')
    # plt.show()
    # plot_scatter(results, schemes)


def plot_scatter(results, schemes):
    x_min = 1e15
    x_max = 0
    y_min = 1e15
    y_max = 0
    header_sample = {}
    proof = {}
    for scheme in schemes:
        header_sample[scheme] = (results['header' + '_' + scheme] +
                                 results['sampling_bytes' + '_' + scheme]) / KB
        proof[scheme] = (results['incorrect_coding_proof' + '_' + scheme]) / KB
        plt.plot((results['incorrect_coding_proof' + '_' + scheme]) / KB,
                 (results['header' + '_' + scheme] +
                  results['sampling_bytes' + '_' + scheme]) / KB,
                 label=scheme, linewidth=3,
                 marker='o', markerfacecolor='w', markersize=8)
        x_min = np.min([x_min, np.min(results['incorrect_coding_proof' + '_' +
                                              scheme])])
        x_max = np.max([x_max, np.max(results['incorrect_coding_proof' + '_' +
                                              scheme])])
        y_min = np.min([y_min, np.min(results['header' + '_' + scheme] +
                                      results['sampling_bytes' + '_' +
                                              scheme])])
        y_max = np.max([y_max, np.max(results['header' + '_' + scheme] +
                                      results['sampling_bytes' + '_' +
                                              scheme])])
    axis_min = np.min([x_min, y_min]) / KB / 2
    axis_max = np.max([x_max, y_max]) / KB * 1.2
    plt.xlim((axis_min, axis_max))
    plt.ylim((axis_min, y_max * 1.1 / KB))
    plt.plot([axis_min, axis_max], [axis_min, axis_max], color='r',
             label='x=y')
    plt.plot([proof[s] for s in schemes],
             [header_sample[s] for s in schemes], color='c')
    for x, y, size in zip(proof['2drs'],
                          header_sample['2drs'], block_size_range):
        plt.text(x * 0.9, y * 1.2, str(int(size / MB)) + 'MB')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('Failed-parity proof size (KB)', fontsize=16)
    plt.ylabel('Header + sampling cost (KB)', fontsize=16)
    plt.tight_layout()
    plt.show()


def myPlot(x_vectors, y_vectors, colors, data_labels, xlabel, ylabel, title,
           scale='log'):
    for x_vec, y_vec, label, c in zip(x_vectors,
                                      y_vectors, data_labels, colors):
        if scale == 'log':
            plt.loglog(x_vec, y_vec, label=label, color=c, linewidth=3,
                       marker='o', markersize=10, markerfacecolor='w')
        else:
            plt.plot(x_vec, y_vec, label=label, color=c, linewidth=3,
                     marker='o', markersize=10, markerfacecolor='w')
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    pass


def block_size_vs_cost_cross(block_size_range, data_symbol_size_range,
                             batch_factor, hash_size, code_rate,
                             stopping_ratio_2drs, stopping_ratio_spar,
                             target_prob, parity_size):
    # 2D-RS always use F_256.
    keywords = ['header_2drs', 'sampling_bytes_2drs',
                'incorrect_coding_proof_2drs', 'total_cost_2drs',
                'header_spar', 'sampling_bytes_spar',
                'incorrect_coding_proof_spar', 'total_cost_spar']

    results = {}
    for k in keywords:
        results[k] = []

    for b, d in zip(block_size_range, data_symbol_size_range):
        costs = cost(b, d, batch_factor, hash_size, code_rate,
                     stopping_ratio_2drs, stopping_ratio_spar,
                     target_prob, parity_size)
        for k, c in zip(keywords[:4], costs[:4]):
            results[k].append(c)
        # recompute for SPAR, as SPAR doesn't have to increase data symbol size
        costs = cost(b, data_symbol_size_range[0],
                     batch_factor, hash_size, code_rate,
                     stopping_ratio_2drs, stopping_ratio_spar,
                     target_prob, parity_size)
        for k, c in zip(keywords[4:], costs[4:]):
            results[k].append(c)

    for k in keywords:
        results[k] = np.array(results[k])

    # compute when 2D-RS has to use a GF > 256
    N_threshold = 65536
    for idx in range(len(block_size_range)):
        block_size_threshold = \
            N_threshold * data_symbol_size_range[idx] * code_rate
        if block_size_range[idx] > block_size_threshold:
            break

    block_size_range = np.array(block_size_range) / MB
    titles = ['header', 'sampling_bytes', 'incorrect_coding_proof', 'total_cost']
    schemes = ['2drs', 'spar']

    # plot absolute costs
    # x_vectors = [block_size_range] * 2 + [block_size_range[idx:]]
    # x_label = 'block size (MB)'
    # colors = ['b', 'g', 'r']
    # for title in titles:
    #     y_vectors = [results[title + '_' + s] / KB for s in schemes] + \
    #         [results[title + '_' + schemes[0]][idx:] / KB]
    #     labels = schemes + [None]
    #     title = title + ' size'
    #     y_label = title + ' (kB)'
    #     myPlot(x_vectors, y_vectors, colors, labels, x_label, y_label, title,
    #            scale='log')

    # plot header + sampling cost
    x_vector = block_size_range
    y_vector_2drs = results['header_2drs'] + results['sampling_bytes_2drs']
    y_vector_spar = results['header_spar'] + results['sampling_bytes_spar']
    plt.semilogy(x_vector, y_vector_2drs / KB, label='2D-RS', marker='o')
    plt.semilogy(x_vector, y_vector_spar / KB, label='SPAR', marker='s')
    plt.xlabel('block size (MB)')
    plt.ylabel('header + sampling (KB)')
    plt.grid()
    plt.legend(loc='best')
    plt.title('header + sampling cost comparison')
    plt.show()

    # plot incorrect coding cost
    x_vector = block_size_range
    y_vector_2drs = results['incorrect_coding_proof_2drs']
    y_vector_spar = results['incorrect_coding_proof_spar']
    plt.semilogy(x_vector, y_vector_2drs / KB, label='2D-RS', marker='o')
    plt.semilogy(x_vector, y_vector_spar / KB, label='SPAR', marker='s')
    plt.xlabel('block size (MB)')
    plt.ylabel('proof size (KB)')
    plt.grid()
    plt.legend(loc='best')
    plt.title('incorrect-coding proof size comparison')
    plt.show()

    # # plot costs normalized to block size
    # x_label = 'block size (MB)'
    # colors = ['b', 'g', 'r']
    # for title in titles:
    #     y_vectors = [results[title + '_' + s] / block_size_range / MB
    #                  for s in schemes] + \
    #                 [results[title + '_' + schemes[0]][idx:] /
    #                  block_size_range[idx:] / MB]
    #     labels = schemes + [None]
    #     title = title + ' to block size ratio'
    #     y_label = title
    #     myPlot(x_vectors, y_vectors, colors, labels, x_label, y_label, title,
    #            scale='log')

    # plot_scatter(results, schemes)


def num_samples(target_prob, stopping_ratio, N):
    K_hide = int(stopping_ratio * N)
    prob = 1
    s = 1
    while True:
        prob *= ((N - K_hide) / N)
        if prob <= target_prob:
            return s
        s += 1
        N -= 1


# block_size_range = 2 ** np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
block_size_range = 2 ** np.array([20, 22, 24, 26])
data_symbol_size = 1024
data_symbol_size_range = [data_symbol_size] * len(block_size_range)
batch_factor = 8
hash_size = 32
code_rate = 0.25
spar_header_size = 1
stopping_ratio_2drs = 0.25
stopping_ratio_spar_strong = 0.124
stopping_ratio_spar_weak = 0.47
target_prob = 0.01
parity_size = 8


block_size_vs_cost(block_size_range, data_symbol_size_range, batch_factor,
                   hash_size, code_rate,
                   spar_header_size, stopping_ratio_2drs,
                   stopping_ratio_spar_strong, stopping_ratio_spar_weak,
                   target_prob, parity_size)


# def dss_adaptive(block_size, code_rate, default_dss=256, N_thres=65536):
#     N = (block_size / default_dss) / code_rate
#     if N <= N_thres:
#         return default_dss
#     return int(block_size / code_rate / N_thres)


# data_symbol_size_range = [dss_adaptive(b, code_rate, data_symbol_size)
#                           for b in block_size_range]
# block_size_vs_cost_cross(block_size_range, data_symbol_size_range,
#                          batch_factor,
#                          hash_size, code_rate, stopping_ratio_2drs,
#                          stopping_ratio_spar, target_prob, parity_size)
