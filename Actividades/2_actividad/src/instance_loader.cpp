#include "instance_loader.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::string InstanceLoader::trim(const std::string &s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) start++;
    auto end = s.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
}

std::vector<Item> InstanceLoader::loadItems(const std::string &path) {
    std::vector<Item> items;
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        Item item{0, 0, 0, 0, ""};

        std::getline(ss, token, ',');
        item.id = std::stoi(trim(token));
        std::getline(ss, token, ',');
        item.value = std::stof(trim(token));
        std::getline(ss, token, ',');
        item.weight = std::stof(trim(token));
        std::getline(ss, token, ',');
        item.volume = std::stof(trim(token));
        std::getline(ss, token, ',');
        item.category = trim(token);

        items.push_back(item);
    }
    return items;
}

CategoryMap InstanceLoader::loadCategoryRules(const std::string &path) {
    CategoryMap rules;
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        CategoryRule rule{0, 0};
        std::string category;

        std::getline(ss, token, ',');
        category = trim(token);
        std::getline(ss, token, ',');
        rule.min = std::stoi(trim(token));
        std::getline(ss, token, ',');
        rule.max = std::stoi(trim(token));

        rules[category] = rule;
    }
    return rules;
}

std::vector<Incompatibility>
InstanceLoader::loadIncompatibilities(const std::string &path) {
    std::vector<Incompatibility> incomp;
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        Incompatibility inc{0, 0};

        std::getline(ss, token, ',');
        inc.id_a = std::stoi(trim(token));
        std::getline(ss, token, ',');
        inc.id_b = std::stoi(trim(token));

        incomp.push_back(inc);
    }
    return incomp;
}

DependencyMap InstanceLoader::loadDependencies(const std::string &path) {
    DependencyMap deps;
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        int id_item, id_required;

        std::getline(ss, token, ',');
        id_item = std::stoi(trim(token));
        std::getline(ss, token, ',');
        id_required = std::stoi(trim(token));

        deps.push_back({id_item, id_required});
    }
    return deps;
}

KnapsackConfig InstanceLoader::loadKnapsackConfig(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return {1000.0f, 500.0f};
    }

    std::string line;
    std::getline(file, line); // skip header
    std::getline(file, line);

    if (line.empty()) {
        return {1000.0f, 500.0f};
    }

    std::stringstream ss(line);
    std::string token;
    KnapsackConfig config;

    std::getline(ss, token, ',');
    config.max_weight = std::stof(trim(token));
    std::getline(ss, token, ',');
    config.max_volume = std::stof(trim(token));

    return config;
}

PenaltyConfig InstanceLoader::loadPenaltyConfig(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
    }

    PenaltyConfig config{0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::string key;
        float value;

        std::getline(ss, key, ',');
        std::getline(ss, token, ',');
        value = std::stof(trim(token));

        key = trim(key);
        if (key == "peso_exceso")
            config.alpha = value;
        else if (key == "volumen_exceso")
            config.beta = value;
        else if (key == "incompatibilidad")
            config.delta = value;
        else if (key == "dependencia")
            config.epsilon = value;
        else if (key == "categoria")
            config.gamma = value;
    }

    return config;
}

Instance InstanceLoader::load(const std::string &directory) {
    Instance instance;
    instance.items = loadItems(directory + "/items.csv");
    instance.category_rules =
        loadCategoryRules(directory + "/category_rules.csv");
    instance.incompatibilities =
        loadIncompatibilities(directory + "/incompatibilities.csv");
    instance.dependencies = loadDependencies(directory + "/dependencies.csv");
    instance.knapsack = loadKnapsackConfig(directory + "/knapsack_config.csv");
    instance.penalties = loadPenaltyConfig(directory + "/penalty_config.csv");

    float total_weight = 0.0f, total_volume = 0.0f;
    for (const auto &item : instance.items) {
        instance.max_value += item.value;
        total_weight += item.weight;
        total_volume += item.volume;
    }
    // Pᵢ_max real: exceso máximo posible si se seleccionan todos los ítems
    instance.max_excess_weight = std::max(0.0f, total_weight - instance.knapsack.max_weight);
    instance.max_excess_volume = std::max(0.0f, total_volume - instance.knapsack.max_volume);

    return instance;
}
