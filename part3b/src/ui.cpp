#include <functional>
#include <string>
#include <cstddef>
#include <algorithm>
#include <array>
#include <utility>
#include <cstdio>

#include <gui_base/gui_base.hpp>
#include <ImGuiFileDialog.h>
#include <implot.h>

#include "network.hpp"
#include "learn.hpp"
#include "ui.hpp"
#include "helpers.hpp"

namespace ui {
    static constexpr auto RED = ImVec4(0.9f, 0.65f, 0.65f, 1.0f);

    bool learning_setup(Learn<18, 1>& learn, network::Network<18, 1>& network) {
        static int hidden_layers = 1;
        static std::array<int, 32> hidden_layer_neurons = { 50, 50, 50 };

        bool apply = false;

        if (ImGui::Begin("Learning Setup")) {
            ImGui::Text("Inputs: %lu", network.get_inputs());
            ImGui::Text("Outputs: %lu", network.get_outputs());

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::InputInt("Hidden layers", &hidden_layers)) {
                hidden_layers = std::max(hidden_layers, 1);
                hidden_layers = std::min(hidden_layers, 32);
            }

            ImGui::Spacing();

            for (int i = 0; i < hidden_layers; i++) {
                ImGui::PushID(i);

                if (ImGui::InputInt("##", &hidden_layer_neurons[i])) {
                    hidden_layer_neurons[i] = std::max(hidden_layer_neurons[i], 1);
                }

                ImGui::SameLine();

                ImGui::Text("%d", i);

                ImGui::PopID();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (learn.training_set.loaded && learn.training_set.normalized) {
                if (ImGui::Button("Apply Setup")) {
                    network::HiddenLayers layers;

                    for (int i = 0; i < hidden_layers; i++) {
                        layers.layers.push_back(hidden_layer_neurons[i]);
                    }

                    network.setup(std::move(layers));

                    apply = true;
                }
            } else {
                if (ImGui::Button("Choose training set")) {
                    ui::open_file_browser();
                }
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::InputDouble("Learning rate", &learn.options.learning_rate);
            ImGui::InputDouble("Epsilon", &learn.options.epsilon);
            ImGui::InputScalar("Max epochs", ImGuiDataType_U64, &learn.options.max_epochs);
        }

        ImGui::End();

        return apply;
    }

    Operation learning_process(const Learn<18, 1>& learn) {
        Operation result = Operation::None;

        if (ImGui::Begin("Learning Process")) {
            ImGui::TextColored(RED, "Epoch index: %lu", learn.learning.epoch_index);
            ImGui::TextColored(RED, "Step index: %lu", learn.learning.step_index);
            ImGui::TextColored(RED, "Current error: %f", learn.learning.epoch_error);
            ImGui::Separator();
            ImGui::Text("Learning rate: %f", learn.options.learning_rate);
            ImGui::Text("Epsilon: %f", learn.options.epsilon);
            ImGui::Text("Max epochs: %lu", learn.options.max_epochs);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (learn.is_running()) {
                if (ImGui::Button("Stop")) {
                    result = Operation::Stop;
                }
            } else {
                if (ImGui::Button("Start")) {
                    result = Operation::Start;
                }

                ImGui::SameLine();

                if (ImGui::Button("Reinitialize")) {
                    result = Operation::Reinitialize;
                }

                ImGui::SameLine();

                if (ImGui::Button("Test")) {
                    result = Operation::Test;
                }

                ImGui::SameLine();

                if (ImGui::Button("Execute")) {
                    result = Operation::Execute;
                }
            }
        }

        ImGui::End();

        return result;
    }

    void learning_graph(const Learn<18, 1>& learn) {
        if (ImGui::Begin("Learning Graph")) {
            ImPlot::SetNextAxesToFit();

            if (ImPlot::BeginPlot("Epoch Error", ImVec2(-1.0f, 0.0f), ImPlotAxisFlags_AutoFit)) {
                ImPlot::SetupAxes("Index", "Error");
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0);

                ImPlot::PlotLine(
                    "Epoch Error",
                    learn.learning.error_graph.indices.data(),
                    learn.learning.error_graph.errors.data(),
                    static_cast<int>(learn.learning.error_graph.indices.size())
                );

                ImPlot::EndPlot();
            }
        }

        ImGui::End();
    }

    void training_set(TrainingSet& training_set) {
        static constexpr int MAX_GROUP = 9;
        static int group = 0;

        const std::size_t group_size = training_set.data.size() / (MAX_GROUP + 1);

        if (ImGui::Begin("Training Set")) {
            ImGui::Text("%lu/%lu are for training", training_set.training_instance_count, training_set.data.size());
            ImGui::Spacing();

            if (ImGui::Button("Shuffle")) {
                training_set.shuffle();
            }

            ImGui::SameLine();

            if (ImGui::Button("Normalize")) {
                training_set.normalize();
            }

            ImGui::Spacing();

            if (ImGui::SmallButton("<")) {
                group = std::max(group - 1, 0);
            }

            ImGui::SameLine();

            if (ImGui::SmallButton(">")) {
                group = std::min(group + 1, MAX_GROUP);
            }

            ImGui::Spacing();

            if (ImGui::BeginTable("Training", 20, ImGuiTableFlags_Borders)) {
                ImGui::TableSetupColumn("Index");
                ImGui::TableSetupColumn("Class");
                ImGui::TableSetupColumn("Current assets");
                ImGui::TableSetupColumn("Cost of goods sold");
                ImGui::TableSetupColumn("Depreciation and amortization");
                ImGui::TableSetupColumn("Financial performance");
                ImGui::TableSetupColumn("Inventory");
                ImGui::TableSetupColumn("Net income");
                ImGui::TableSetupColumn("Total receivables");
                ImGui::TableSetupColumn("Market value");
                ImGui::TableSetupColumn("Net sales");
                ImGui::TableSetupColumn("Total assets");
                ImGui::TableSetupColumn("Total long-term debt");
                ImGui::TableSetupColumn("EBIT");
                ImGui::TableSetupColumn("Gross profit");
                ImGui::TableSetupColumn("Total current liabilities");
                ImGui::TableSetupColumn("Retained earnings");
                ImGui::TableSetupColumn("Total revenue");
                ImGui::TableSetupColumn("Total liabilities");
                ImGui::TableSetupColumn("Total operating expenses");
                ImGui::TableHeadersRow();

                const std::size_t begin {group * group_size};
                const std::size_t end {group == MAX_GROUP ? training_set.data.size() : (group + 1) * group_size};

                for (std::size_t i {1}, j {begin}; j < end; j++) {
                    const Instance& instance = training_set.data[j];

                    ImGui::TableNextColumn();
                    ImGui::Text("%lu", i);

                    if (i > training_set.training_instance_count) {
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(45, 45, 55, 255));
                    }

                    i++;

                    if (training_set.normalized) {
                        ImGui::TableNextColumn();
                        ImGui::Text("%f", instance.classification);
                    } else {
                        ImGui::TableNextColumn();
                        ImGui::Text("%s", instance.classification == 1.0 ? "alive" : "failed");
                    }

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.current_assets);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.cost_of_goods_sold);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.depreciation_and_amortization);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.financial_performance);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.inventory);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.net_income);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.total_receivables);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.market_value);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.net_sales);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.total_assets);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.total_long_term_debt);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.earnings_before_interest_and_taxes);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.gross_profit);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.total_current_liabilities);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.retained_earnings);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.total_revenue);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.total_liabilities);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", instance.total_operating_expenses);
                }

                ImGui::EndTable();
            }
        }

        ImGui::End();
    }

    void open_file_browser() {
        ImGuiFileDialog::Instance()->OpenDialog(
            "FileDialog",
            "Choose File",
            ".csv,.txt",
            ".",
            1,
            nullptr,
            ImGuiFileDialogFlags_Modal
        );
    }

    void file_browser(const std::function<void(const std::string&)>& callback) {
        if (ImGuiFileDialog::Instance()->Display("FileDialog", 32, ImVec2(768, 432))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                const auto file_path = ImGuiFileDialog::Instance()->GetFilePathName();
                callback(file_path);
            }

            ImGuiFileDialog::Instance()->Close();
        }
    }

    bool testing(const Learn<18, 1>& learn, const network::Network<18, 1>& network) {
        bool back = false;

        if (ImGui::Begin("Testing")) {
            ImGui::Text("Trained for %lu epochs", learn.learning.epoch_index + 1);
            ImGui::Text("Last epoch error: %f", learn.learning.epoch_error);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            static double test_result {0.0};

            if (ImGui::Button("Test")) {
                test_result = learn.test(network);
            }

            ImGui::SameLine();

            if (ImGui::Button("Go back")) {
                back = true;
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextColored(RED, "Test result: %f %%", test_result);

            ImGui::Spacing();

            if (ImGui::BeginTable("Tests", 22, ImGuiTableFlags_Borders)) {
                ImGui::TableSetupColumn("Index");
                ImGui::TableSetupColumn("RESULT");
                ImGui::TableSetupColumn("OUTPUT");
                ImGui::TableSetupColumn("Class");
                ImGui::TableSetupColumn("X1");
                ImGui::TableSetupColumn("X2");
                ImGui::TableSetupColumn("X3");
                ImGui::TableSetupColumn("X4");
                ImGui::TableSetupColumn("X5");
                ImGui::TableSetupColumn("X6");
                ImGui::TableSetupColumn("X7");
                ImGui::TableSetupColumn("X8");
                ImGui::TableSetupColumn("X9");
                ImGui::TableSetupColumn("X10");
                ImGui::TableSetupColumn("X11");
                ImGui::TableSetupColumn("X12");
                ImGui::TableSetupColumn("X13");
                ImGui::TableSetupColumn("X14");
                ImGui::TableSetupColumn("X15");
                ImGui::TableSetupColumn("X16");
                ImGui::TableSetupColumn("X17");
                ImGui::TableSetupColumn("X18");
                ImGui::TableHeadersRow();

                for (std::size_t i {1}; const Test& test : learn.testing.tests) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%lu", i);

                    if (!test.passed) {
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(45, 45, 55, 255));
                    }

                    i++;

                    ImGui::TableNextColumn();
                    ImGui::Text("%s", test.passed ? "pass" : "fail");

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.output);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.classification);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.current_assets);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.cost_of_goods_sold);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.depreciation_and_amortization);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.financial_performance);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.inventory);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.net_income);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.total_receivables);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.market_value);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.net_sales);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.total_assets);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.total_long_term_debt);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.earnings_before_interest_and_taxes);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.gross_profit);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.total_current_liabilities);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.retained_earnings);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.total_revenue);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.total_liabilities);

                    ImGui::TableNextColumn();
                    ImGui::Text("%f", test.instance.total_operating_expenses);
                }

                ImGui::EndTable();
            }
        }

        ImGui::End();

        return back;
    }

    bool executing(const network::Network<18, 1>& network) {
        static std::array<double, 18> user_inputs {};
        static std::array<double, 18> inputs {};
        static std::array<double, 1> outputs {};

        bool back = false;

        if (ImGui::Begin("Executing")) {
            ImGui::Text("All the attributes represent sums of money");
            ImGui::Spacing();

            ImGui::InputDouble("Current assets", user_inputs.data() + 0);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("All the assets of a company that are expected to be sold or used as a result of standard business operations over the next year");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Cost of goods sold", user_inputs.data() + 1);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The total amount a company paid as a cost directly related to the sale of products");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Depreciation and amortization", user_inputs.data() + 2);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("Depreciation refers to the loss of value of a tangible fixed asset over time (such as property, machinery, buildings, and plant); amortization refers to the loss of value of intangible assets over time");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Financial performance", user_inputs.data() + 3);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("It is a measure of a company's overall financial performance, serving as an alternative to net income");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Inventory", user_inputs.data() + 4);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The accounting of items and raw materials that a company either uses in production or sells");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Net income", user_inputs.data() + 5);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The overall profitability of a company after all expenses and costs have been deducted from total revenue");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Total receivables", user_inputs.data() + 6);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The balance of money due to a firm for goods or services delivered or used but not yet paid for by customers");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Market value", user_inputs.data() + 7);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The price of an asset in a marketplace. In this dataset, it refers to the market capitalization since companies are publicly traded in the stock market");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Net sales", user_inputs.data() + 8);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The sum of a company's gross sales minus its returns, allowances, and discounts");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Total assets", user_inputs.data() + 9);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("All the assets, or items of value, a business owns");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Total long-term debt", user_inputs.data() + 10);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("A company's loans and other liabilities that will not become due within one year of the balance sheet date");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("EBIT", user_inputs.data() + 11);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("Earnings before interest and taxes");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Gross profit", user_inputs.data() + 12);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The profit a business makes after subtracting all the costs that are related to manufacturing and selling its products or services");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Total current liabilities", user_inputs.data() + 13);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The sum of accounts payable, accrued liabilities, and taxes such as Bonds payable at the end of the year, salaries, and commissions remaining");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Retained earnings", user_inputs.data() + 14);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The amount of profit a company has left over after paying all its direct costs, indirect costs, income taxes, and its dividends to shareholders");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Total revenue", user_inputs.data() + 15);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The amount of income that a business has made from all sales before subtracting expenses; it may include interest and dividends from investments");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Total liabilities", user_inputs.data() + 16);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The combined debts and obligations that the company owes to outside parties");
                    ImGui::EndTooltip();
                }
            }

            ImGui::InputDouble("Total operating expenses", user_inputs.data() + 17);
            if (ImGui::IsItemHovered()) {
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("The expenses a business incurs through its normal business operations");
                    ImGui::EndTooltip();
                }
            }

            ImGui::Spacing();

            const char* result = network::functions::binary(outputs[0]) == 1.0 ? "alive" : "failed";

            ImGui::TextColored(RED, "Result: %f (%s)", outputs[0], result);

            ImGui::Spacing();

            if (ImGui::Button("Execute")) {
                Instance instance;
                instance.current_assets = user_inputs[0];
                instance.cost_of_goods_sold = user_inputs[1];
                instance.depreciation_and_amortization = user_inputs[2];
                instance.financial_performance = user_inputs[3];
                instance.inventory = user_inputs[4];
                instance.net_income = user_inputs[5];
                instance.total_receivables = user_inputs[6];
                instance.market_value = user_inputs[7];
                instance.net_sales = user_inputs[8];
                instance.total_assets = user_inputs[9];
                instance.total_long_term_debt = user_inputs[10];
                instance.earnings_before_interest_and_taxes = user_inputs[11];
                instance.gross_profit = user_inputs[12];
                instance.total_current_liabilities = user_inputs[13];
                instance.retained_earnings = user_inputs[14];
                instance.total_revenue = user_inputs[15];
                instance.total_liabilities = user_inputs[16];
                instance.total_operating_expenses = user_inputs[17];

                normalize_instance(instance);

                inputs[0] = instance.current_assets;
                inputs[1] = instance.cost_of_goods_sold;
                inputs[2] = instance.depreciation_and_amortization;
                inputs[3] = instance.financial_performance;
                inputs[4] = instance.inventory;
                inputs[5] = instance.net_income;
                inputs[6] = instance.total_receivables;
                inputs[7] = instance.market_value;
                inputs[8] = instance.net_sales;
                inputs[9] = instance.total_assets;
                inputs[10] = instance.total_long_term_debt;
                inputs[11] = instance.earnings_before_interest_and_taxes;
                inputs[12] = instance.gross_profit;
                inputs[13] = instance.total_current_liabilities;
                inputs[14] = instance.retained_earnings;
                inputs[15] = instance.total_revenue;
                inputs[16] = instance.total_liabilities;
                inputs[17] = instance.total_operating_expenses;

                network.run(inputs.data(), outputs.data());
            }

            ImGui::SameLine();

            if (ImGui::Button("Go back")) {
                back = true;
            }
        }

        ImGui::End();

        return back;
    }
}
