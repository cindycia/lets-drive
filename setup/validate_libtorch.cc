#include <iostream>
#include <iomanip>
#include <torch/torch.h>

int main() {

    const int64_t input_size   = 1;
    const int64_t output_size  = 1;

    const double learning_rate = 0.01;
    const size_t num_epochs    = 150;

    // Random dataset
    auto x_train = torch::randint(0, 5, {20, 1});
    auto y_train = torch::randint(0, 5, {20, 1});

    // Linear regression
    torch::nn::Linear model(input_size, output_size);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    std::cout << std::fixed << std::setprecision(5);

    // Train loop
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {

        // Forward
        auto output = model(x_train);
        auto loss = torch::nn::functional::mse_loss(output, y_train);

        // Backward and gradient step
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout << "Epoch: " << epoch + 1 << "/" << num_epochs
                  << " | Loss: " << loss.item<double>() << std::endl;
    }
    return 0;
}
