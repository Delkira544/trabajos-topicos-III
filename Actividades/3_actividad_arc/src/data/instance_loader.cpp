#include "data/instance_loader.hpp"
#include "config/constants.hpp"
#include "rapidcsv.h"
#include "types.hpp"

namespace ga::data
{
  KnapsackInstance InstanceLoader::load_instance(const std::string& file_path)
  {
    KnapsackInstance instance;
    instance.items = load_items(file_path + "/items.csv", instance);
    instance.category_rules =
      load_category_rules(file_path + "/category_rules.csv");
    instance.incompatibility_rules =
      load_incompatibility_rules(file_path + "/incompatibilities.csv");
    instance.dependency_rules =
      load_dependency_rules(file_path + "/dependencies.csv");
    return instance;
  }

  std::vector<Item> InstanceLoader::load_items(const std::string& file_path,
                                               KnapsackInstance& instance)
  {
    try
    {
      std::vector<Item> items;
      rapidcsv::Document doc(file_path);
      size_t n_rows       = doc.GetRowCount();
      double total_weight = 0;
      double total_volume = 0;
      double total_value  = 0;
      items.reserve(n_rows);
      for (size_t i = 0; i < n_rows; ++i)
      {
        Item item;
        // Asumimos que el CSV tiene columnas: id, value, weight, volume,
        // category
        item.id       = doc.GetCell<int>("id", i);
        item.value    = doc.GetCell<float>("value", i);
        item.weight   = doc.GetCell<float>("weight", i);
        item.volume   = doc.GetCell<float>("volume", i);
        item.category = doc.GetCell<std::string>("category", i);

        total_weight += item.weight;
        total_volume += item.volume;
        total_value += item.value;

        items.push_back(item);
      }

      instance.max_volume =
        total_volume * Config::GeneticAlgorithm::PERCENTAGE_CAPACITY;
      instance.max_weight =
        total_weight * Config::GeneticAlgorithm::PERCENTAGE_CAPACITY;
      instance.max_value = total_value;

      return items;
    } catch (const std::exception& e)
    {
      throw std::runtime_error("Failed to load items from " + file_path + ": " + e.what());
    }
  }

  std::unordered_map<std::string, CategoryRule>
  InstanceLoader::load_category_rules(const std::string& file_path)
  {
    try
    {
      std::unordered_map<std::string, CategoryRule> category_rules;
      rapidcsv::Document doc(file_path);
      size_t n_rows = doc.GetRowCount();
      for (size_t i = 0; i < n_rows; ++i)
      {
        std::string category     = doc.GetCell<std::string>("category", i);
        int min                  = doc.GetCell<int>("min", i);
        int max                  = doc.GetCell<int>("max", i);
        category_rules[category] = {min, max};
      }
      return category_rules;
    } catch (const std::exception& e)
    {
      throw std::runtime_error("Failed to load category rules from " + file_path + ": " + e.what());
    }
  }

  std::vector<IncompatibilityRule> InstanceLoader::load_incompatibility_rules(
    const std::string& file_path)
  {
    try
    {
      std::vector<IncompatibilityRule> incompatibility_rules;
      rapidcsv::Document doc(file_path);
      size_t n_rows = doc.GetRowCount();
      incompatibility_rules.reserve(n_rows);
      for (size_t i = 0; i < n_rows; ++i)
      {
        IncompatibilityRule rule;
        rule.item_id_a = doc.GetCell<int>("id_item_a", i);
        rule.item_id_b = doc.GetCell<int>("id_item_b", i);
        incompatibility_rules.push_back(rule);
      }
      return incompatibility_rules;
    } catch (const std::exception& e)
    {
      throw std::runtime_error("Failed to load incompatibility rules from " + file_path + ": " + e.what());
    }
  }

  std::vector<DependencyRule> InstanceLoader::load_dependency_rules(
    const std::string& file_path)
  {
    try
    {
      std::vector<DependencyRule> dependency_rules;
      rapidcsv::Document doc(file_path);
      size_t n_rows = doc.GetRowCount();
      dependency_rules.reserve(n_rows);
      for (size_t i = 0; i < n_rows; ++i)
      {
        DependencyRule rule;
        rule.item_id_a = doc.GetCell<int>("id_item", i);
        rule.item_id_b = doc.GetCell<int>("id_dependency", i);
        dependency_rules.push_back(rule);
      }
      return dependency_rules;
    } catch (const std::exception& e)
    {
      throw std::runtime_error("Failed to load dependency rules from " + file_path + ": " + e.what());
    }
  }
} // namespace ga::data
