#ifndef INSTANCE_LOADER_HPP
#define INSTANCE_LOADER_HPP

#include "genetic_algorithm.hpp"
#include <string>

// =============================================================================
// InstanceLoader — carga de los CSV de una instancia
// =============================================================================
// Lee los 6 archivos CSV que componen una instancia de la mochila extendida
// y los ensambla en una estructura `Instance` lista para el AG.
//
// Archivos esperados en `directory/`:
//   items.csv             id, valor, peso, volumen, categoria
//   category_rules.csv    categoria, minimo, maximo
//   incompatibilities.csv id_item_a, id_item_b
//   dependencies.csv      id_item, id_requerido
//   knapsack_config.csv   max_weight, max_volume
//   penalty_config.csv    penalty_type, value
//
// Todos los `loadX()` lanzan `std::runtime_error` si el archivo no existe o
// no se puede abrir (excepto `loadKnapsackConfig` y `loadPenaltyConfig`, que
// retornan defaults razonables para mantener el AG ejecutable sin esos CSV).
// =============================================================================

class InstanceLoader {
  public:
    /**
     * @brief Carga la instancia completa desde un directorio.
     *
     * Además de leer los archivos, precalcula:
     *   - `instance.max_value`         = Σ valores de todos los ítems
     *   - `instance.max_excess_weight` = Σ pesos − max_weight (≥ 0)
     *   - `instance.max_excess_volume` = Σ volúmenes − max_volume (≥ 0)
     */
    static Instance load(const std::string &directory);

  private:
    // Cada loader es responsable de uno de los 6 CSV.
    static std::vector<Item> loadItems(const std::string &path);
    static CategoryMap loadCategoryRules(const std::string &path);
    static std::vector<Incompatibility>
    loadIncompatibilities(const std::string &path);
    static DependencyMap loadDependencies(const std::string &path);
    static KnapsackConfig loadKnapsackConfig(const std::string &path);
    static PenaltyConfig loadPenaltyConfig(const std::string &path);

    /// Helper: elimina espacios en blanco al inicio y al final del string.
    static std::string trim(const std::string &s);
};

#endif
