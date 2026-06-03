#pragma once
#include "types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace ga::data
{
  class InstanceLoader
  {
    public:
    static KnapsackInstance load_instance(const std::string& file_path);

    private:
    static std::vector<Item> load_items(const std::string& file_path,
                                        KnapsackInstance& instance);
    static std::unordered_map<std::string, CategoryRule> load_category_rules(
      const std::string& file_path);
    static std::vector<IncompatibilityRule> load_incompatibility_rules(
      const std::string& file_path);
    static std::vector<DependencyRule> load_dependency_rules(
      const std::string& file_path);
  };

} // namespace ga::data
