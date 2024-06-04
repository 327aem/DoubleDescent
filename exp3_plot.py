from ast import literal_eval
import matplotlib.pyplot as plt

log_dir = "./log"
result_types = ["test_acc", "test_loss", 
                "train_acc", "train_loss", 
                ]

plot_color = {
    2:"#caf0f8",
    3:"#90e0ef",
    4:"#48cae4",
    5:"#00b4d8",
    6:"#0096c7",
    7:"#0077b6",
    8:"#023e8a",
    9:"#03045e",
    10:"#000000",
}
plot_marker = {
    2:".",
    3:"^",
    4:"s",
    5:"p",
    6:"H",
    7:"*",
    8:"8",
    9:"X",
    10:"P",
}

plot_line_styles = {
    2:"-",
    3:":",
    4:"-",
    5:":",
    6:"-",
    7:":",
    8:"-",
    9:":",
    10:"-",
}

for result_type in result_types:
    plt.figure(figsize=(10, 6))

    for num_noised_class in num_noised_classes:

        f = open(f"{log_dir}/{result_type}_{{'model'_ 'resnet', 'dataset'_ 'Cifar10', 'batch_size'_ 128, 'learning_rate'_ 0.0001, 'label_noise'_ 0.15, 'num_noised_class'_ {num_noised_class}, 'img_noise'_ None}}.txt", 'r')
        results = f.readline()
        f.close()

        if results.count(",") == 64:
            results = f"{{{results[:-3]}}}"
            results_dict = literal_eval(results)
            assert isinstance(results_dict, dict)

            result_list = sorted(results_dict.items(), key = lambda x : x[0])
            K_values, performances = zip(*result_list)

            plt.plot(K_values, performances, label=f"num_noised_class:{num_noised_class}", color=plot_color[num_noised_class], marker=plot_marker[num_noised_class], linestyle=plot_line_styles[num_noised_class])
            
        else:
            raise Exception(f"The experiment results are not complete. Given results have {results.count(',')} results, not 64. File name:", f"{log_dir}/{result_type}_{{'model'_ 'resnet', 'dataset'_ 'Cifar10', 'batch_size'_ 128, 'learning_rate'_ 0.0001, 'label_noise'_ 0.15, 'num_noised_class'_ {num_noised_class}, 'img_noise'_ None}}.txt")

    plt.title(f'Experimental Result ({result_type})')
    plt.xlabel('K')
    plt.ylabel(f'{result_type}')
    # plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(f"fig/exp3_{result_type}.png", dpi=300)

    if result_type[-4:] == "_acc":
        plt.figure(figsize=(10, 6))

        for num_noised_class in num_noised_classes:

            f = open(f"{log_dir}/{result_type}_{{'model'_ 'resnet', 'dataset'_ 'Cifar10', 'batch_size'_ 128, 'learning_rate'_ 0.0001, 'label_noise'_ 0.15, 'num_noised_class'_ {num_noised_class}, 'img_noise'_ None}}.txt", 'r')
            results = f.readline()
            f.close()

            if results.count(",") == 64:
                results = f"{{{results[:-3]}}}"
                results_dict = literal_eval(results)
                assert isinstance(results_dict, dict)

                result_list = sorted(results_dict.items(), key = lambda x : x[0])
                K_values, performances = zip(*result_list)
                errors = [1-i for i in performances]

                plt.plot(K_values, errors, label=f"num_noised_class:{num_noised_class}", color=plot_color[num_noised_class], marker=plot_marker[num_noised_class], linestyle=plot_line_styles[num_noised_class])
                
            else:
                raise Exception(f"The experiment results are not complete. Given results have {results.count(',')} results, not 64. File name:", f"{log_dir}/{result_type}_{{'model'_ 'resnet', 'dataset'_ 'Cifar10', 'batch_size'_ 128, 'learning_rate'_ 0.0001, 'label_noise'_ 0.15, 'num_noised_class'_ {num_noised_class}, 'img_noise'_ None}}.txt")

        result_type = result_type.replace("_acc", "_error")
        plt.title(f'Experimental Result ({result_type})')
        plt.xlabel('K')
        plt.ylabel(f'{result_type}')
        # plt.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig(f"fig/exp1_{result_type}.png", dpi=300)