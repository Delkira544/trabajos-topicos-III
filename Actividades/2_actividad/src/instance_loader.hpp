#ifndef INSTANCE_LOADER_HPP
#define INSTANCE_LOADER_HPP

#include "genetic_algorithm.hpp"
#include <string>

class InstanceLoader {
public:
    static Instance load(const std::string& directory);

private:
    static std::vector<Item> loadItems(const std::string& path);
    static CategoryMap loadCategoryRules(const std::string& path);
    static std::vector<Incompatibility> loadIncompatibilities(const std::string& path);
    static DependencyMap loadDependencies(const std::string& path);
    static std::string trim(const std::string& s);
};

#endif
