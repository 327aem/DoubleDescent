from ast import literal_eval
import matplotlib.pyplot as plt

log_dir = "./log"
result_types = ["test_acc", "test_loss", "train_acc", "train_loss", "noise_acc", "noise_loss"]
num_noised_classes = [2, 4, 6, 8, 10]

plot_color = {
    2:"#6baed6",
    4:"#4292c6",
    6:"#2171b5",
    8:"#08519c",
    10:"#08306b",
}
plot_marker = {
    2:".",
    4:"s",
    6:"H",
    8:"8",
    10:"P",
}

for result_type in result_types:
    plt.figure(figsize=(10, 6))

    for num_noised_class in num_noised_classes:

        f = open(f"{log_dir}/{result_type}_{{'model': 'resnet', 'dataset': 'Cifar10', 'batch_size': 128, 'learning_rate': 0.0001, 'label_noise': 0.15, 'num_noised_class': {num_noised_class}, 'img_noise': None}}.txt", 'r')
        results = f.readline()
        f.close()

        if results.count(",") == 64:
            results = f"{{{results[:-3]}}}"
            results_dict = literal_eval(results)
            assert isinstance(results_dict, dict)

            result_list = sorted(results_dict.items(), key = lambda x : x[0])
            K_values, performances = zip(*result_list)

            plt.plot(K_values, performances, label=f"num_noised_class:{num_noised_class}", color=plot_color[num_noised_class], marker=plot_marker[num_noised_class])

        else:
            raise Exception(f"The experiment results are not complete. Given results have {results.count(',')} results, not 64.")

    plt.title(f'Experimental Result ({result_type})')
    plt.xlabel('K')
    plt.ylabel(f'{result_type}')
    # plt.grid(True)
    plt.legend()
    plt.show()