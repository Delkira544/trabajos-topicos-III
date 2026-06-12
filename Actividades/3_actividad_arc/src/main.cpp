#include "CLI11.hpp"
#include "cli/app_parser.hpp"
#include "data/instance_loader.hpp"
#include "ga/SolverFactory.hpp"
#include "ga/solvers/CUDABasic.cuh"
#include "types.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char** argv)
{
    CLI::App app;
    AppConfig config;
    configurar_cli(app, config);
    CLI11_PARSE(app, argc, argv);

    // ── Cargar instancia ─────────────────────────────────────────────
    KnapsackInstance instance =
        ga::data::InstanceLoader::load_instance(config.instance);

    std::cout << "\n========== Problem Instance ==========\n";
    std::cout << "Instance:   " << config.instance       << "\n";
    std::cout << "Items:      " << instance.items.size() << "\n";
    std::cout << "Max weight: " << instance.max_weight   << "\n";
    std::cout << "Max volume: " << instance.max_volume   << "\n";

    std::cout << "\n========== Algorithm Configuration ==========\n";
    std::cout << "Variant:        " << config.variant         << "\n";
    std::cout << "Threads:        " << config.threads         << "\n";
    std::cout << "Seed:           " << config.seed            << "\n";
    std::cout << "Crossover rate: " << config.crossover_rate  << "\n";
    std::cout << "Mutation rate:  " << config.mutation_rate   << "\n";
    std::cout << "Tournament size:" << config.tournament_size << "\n";
    std::cout << "Population size:" << config.population_size << "\n";
    std::cout << "Generations:    " << config.num_generations << "\n";
    std::cout << "Penalty weight: " << config.penalty_weight  << "\n";
    std::cout << "Penalty volume: " << config.penalty_volume  << "\n";
    std::cout << "Penalty category: " << config.penalty_category << "\n";
    std::cout << "Penalty incomp: " << config.penalty_incomp  << "\n";
    std::cout << "Penalty dep:    " << config.penalty_dep     << "\n";

    bool is_cuda = (config.variant == "cuda_basic" ||
                    config.variant == "cuda_optimized");

    if (config.variant == "islands_sequential" ||
        config.variant == "islands_parallel") {
        std::cout << "Num islands:      " << config.num_islands        << "\n";
        std::cout << "Migration interval:" << config.migration_interval << "\n";
    }
    if (is_cuda) {
        std::cout << "Block size (CUDA):" << config.block_size << "\n";
    }
    std::cout << "======================================\n\n";

    try {
        SolverFactory::SolverConfig sc;
        sc.num_threads        = config.threads;
        sc.num_islands        = config.num_islands;
        sc.migration_interval = config.migration_interval;
        sc.crossover_rate     = config.crossover_rate;
        sc.mutation_rate      = config.mutation_rate;
        sc.tournament_size    = config.tournament_size;
        sc.seed               = config.seed;
        sc.verbose            = config.verbose;
        sc.block_size         = config.block_size;
        sc.population_size    = config.population_size;
        sc.num_generations    = config.num_generations;
        sc.penalty_weight     = config.penalty_weight;
        sc.penalty_volume     = config.penalty_volume;
        sc.penalty_category   = config.penalty_category;
        sc.penalty_incomp     = config.penalty_incomp;
        sc.penalty_dep        = config.penalty_dep;

        // ── Medir tiempo total con chrono ────────────────────────────
        auto wall_start = std::chrono::high_resolution_clock::now();

        auto solver = SolverFactory::create(config.variant, instance, sc);
        solver->run();

        auto wall_end  = std::chrono::high_resolution_clock::now();
        long wall_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                             wall_end - wall_start).count();

        // ── Resultado principal ───────────────────────────────────────
        Individual best = solver->get_best();

        std::cout << "\n========== Final Result ==========\n";
        std::cout << "Best fitness:        " << best.fitness                     << "\n";
        std::cout << "Feasible:            " << (best.is_valid ? "Yes" : "No")  << "\n";
        std::cout << "Hard feasible:       " << (best.hard_feasible ? "Yes":"No")<< "\n";
        std::cout << "Wall-clock time (ms):" << wall_ms                         << "\n";

        int count = 0;
        for (size_t i = 0; i < best.chromosome.size(); ++i)
            if (best.chromosome[i]) count++;
        std::cout << "Items selected:      " << count << "\n";

        // ── Métricas CUDA (solo si es variante GPU) ───────────────────
        if (is_cuda) {
            // Hacer downcast seguro para acceder a las métricas
            CUDABasic* cuda_solver = dynamic_cast<CUDABasic*>(solver.get());
            if (cuda_solver) {
                long   samples       = cuda_solver->get_timing_samples();
                float  fit_total     = cuda_solver->get_kernel_fitness_ms();
                float  repro_total   = cuda_solver->get_kernel_repro_ms();
                float  h2d_total     = cuda_solver->get_transfer_h2d_ms();
                float  d2h_total     = cuda_solver->get_transfer_d2h_ms();
                float  feasible_pct  = cuda_solver->get_feasible_pct();

                std::cout << "\n========== CUDA Metrics ==========\n";
                std::cout << "Generations measured:          " << samples << "\n";

                if (samples > 0) {
                    std::cout << "Kernel fitness  total (ms):    " << fit_total   << "\n";
                    std::cout << "Kernel fitness  avg/gen (ms):  " << fit_total / samples   << "\n";
                    std::cout << "Kernel repro    total (ms):    " << repro_total << "\n";
                    std::cout << "Kernel repro    avg/gen (ms):  " << repro_total / samples << "\n";
                    std::cout << "Transfer H->D   total (ms):    " << h2d_total   << "\n";
                    std::cout << "Transfer D->H   total (ms):    " << d2h_total   << "\n";
                    float kernel_total   = fit_total + repro_total;
                    float transfer_total = h2d_total + d2h_total;
                    float overhead_pct   = (wall_ms > 0)
                        ? 100.f * transfer_total / (float)wall_ms : 0.f;
                    std::cout << "Kernels total   (ms):          " << kernel_total   << "\n";
                    std::cout << "Transfers total (ms):          " << transfer_total << "\n";
                    std::cout << "Transfer overhead (%):         " << overhead_pct   << "\n";
                }
                std::cout << "Feasible solutions (%):        " << feasible_pct << "\n";
            }
        }

        // ── Métricas de calidad de la historia de fitness ─────────────
        const auto& hist = solver->get_fitness_history();
        if (!hist.empty()) {
            float best_hist = *std::max_element(hist.begin(), hist.end());
            float first     = hist.front();
            std::cout << "\n========== Solution Quality ==========\n";
            std::cout << "Initial best fitness:  " << first     << "\n";
            std::cout << "Final best fitness:    " << best_hist << "\n";
            std::cout << "Improvement:           " << (best_hist - first) << "\n";
            std::cout << "Generations run:       " << hist.size()         << "\n";
        }

        std::cout << "==================================\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}