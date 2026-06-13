#pragma once
#include "string"
#include "unordered_map"
#include "vector"

struct Item
{
  int id;
  float value;
  float weight;
  float volume;
  std::string category;
};

struct CategoryRule
{
  int min;
  int max;
};

struct IncompatibilityRule
{
  int item_id_a;
  int item_id_b;
};

struct DependencyRule
{
  int item_id_a;
  int item_id_b;
};

struct KnapsackInstance
{
  std::vector<Item> items;
  float max_value;
  float max_weight;
  float max_volume;
  std::unordered_map<std::string, CategoryRule> category_rules;
  std::vector<IncompatibilityRule> incompatibility_rules;
  std::vector<DependencyRule> dependency_rules;
};

struct Individual
{
  std::vector<bool> chromosome;
  float fitness;
  bool hard_feasible;
  bool is_valid;
  float penalty;

  Individual() : fitness(0.0f), penalty(0.0f), is_valid(false), hard_feasible(false) {}
};
