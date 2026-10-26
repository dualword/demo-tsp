// https://github.com/dualword/demo-tsp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>

struct Point {
    int id;
    double x;
    double y;
};

struct Individual {
    std::vector<int> chromosome;
    double distance = 0.0;
    double fitness = 0.0;
};

std::vector<Point> parseTSPFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    std::vector<Point> nodes;
    std::string line;
    bool coordSection = false;
    int id = 0;
    while (std::getline(file, line)) {
        if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            coordSection = true;
            continue;
        }
        if (line.find("EOF") != std::string::npos) break;
        if (coordSection) {
            std::stringstream ss(line);
            int fid;
            double x, y;
            if (ss >> fid >> x >> y) {
                nodes.push_back(Point{id++, x, y});
            }
        }
    }
    return nodes;
}

int calculateDistance(const std::vector<int>& chromosome, const std::vector<Point>& nodes) {
    auto getdist = [](const Point& c1, const Point& c2) -> int{
        return std::lround(std::hypot(c1.x - c2.x, c1.y - c2.y));
    };
    int totalDist = 0.0;
    for (size_t i = 0; i < chromosome.size() - 1; ++i) {
        totalDist += getdist(nodes[chromosome[i]], nodes[chromosome[i+1]]);
    }
    totalDist += getdist(nodes[chromosome.back()], nodes[chromosome.front()]);
    return totalDist;
}

class GeneticAlgorithm {
private:
    std::vector<Point> nodes;
    int popSize;
    int tournamentSize;
    int numCities;
    std::vector<Individual> population;
    Individual individual;
    std::mt19937 rng;

    void evaluate(Individual& ind) {
        ind.distance = calculateDistance(ind.chromosome, nodes);
        ind.fitness = 1.0 / (ind.distance);
    }

    Individual selection() {
        std::uniform_int_distribution<int> dist(0, popSize - 1);
        int id = dist(rng);
        for (int i = 1; i < tournamentSize; ++i) {
            int currentIdx = dist(rng);
            if (population[currentIdx].fitness > population[id].fitness) {
                id = currentIdx;
            }
        }
        return population[id];
    }

    Individual crossover(const Individual& p1, const Individual& p2) {
        Individual child;
        child.chromosome.resize(numCities, -1);
        std::vector<bool> buf;
        buf.resize(numCities, false);

        std::uniform_int_distribution<int> dist(0, numCities - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if (cut1 > cut2) std::swap(cut1, cut2);

        for (int i = cut1; i <= cut2; ++i) {
            child.chromosome[i] = p1.chromosome[i];
            buf[p1.chromosome[i]] = true;
        }

        int childIdx = (cut2 + 1) % numCities;
        int parentIdx = (cut2 + 1) % numCities;
        int itemsPlaced = (cut2 - cut1) + 1;

        while (itemsPlaced < numCities) {
            int currentCity = p2.chromosome[parentIdx];
            if (!buf[currentCity]) {
                child.chromosome[childIdx] = currentCity;
                buf[currentCity] = true;
                childIdx = (childIdx + 1) % numCities;
                itemsPlaced++;
            }
            parentIdx = (parentIdx + 1) % numCities;
        }
        return child;
    }

    void mutation(Individual& ind) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < 0.1) {
            std::uniform_int_distribution<int> dist(0, numCities - 1);
            int id1 = dist(rng);
            int id2 = dist(rng);
            if (id1 > id2) std::swap(id1, id2);
            if (id1 != id2) {
                std::reverse(ind.chromosome.begin() + id1, ind.chromosome.begin() + id2 + 1);
            }
        }
    }

public:
    GeneticAlgorithm(std::vector<Point> tspNodes, int populationSize, int tourSize, unsigned int seed)
        : nodes(std::move(tspNodes)), popSize(populationSize), tournamentSize(tourSize), rng(seed) {

        numCities = nodes.size();
        population.resize(popSize);
        std::vector<int> baseChromosome(numCities);
        std::iota(baseChromosome.begin(), baseChromosome.end(), 0);
        individual.distance = std::numeric_limits<double>::max();
        for (int i = 0; i < popSize; ++i) {
            population[i].chromosome = baseChromosome;
            std::shuffle(population[i].chromosome.begin(), population[i].chromosome.end(), rng);
            evaluate(population[i]);
            if (population[i].distance < individual.distance) {
                individual = population[i];
            }
        }
    }

    void evolveOneGeneration() {
        std::vector<Individual> next;
        next.reserve(popSize);
        next.push_back(individual);

        while (next.size() < static_cast<size_t>(popSize)) {
            Individual parent1 = selection();
            Individual parent2 = selection();
            Individual child = crossover(parent1, parent2);
            mutation(child);
            evaluate(child);
            next.push_back(child);
        }

        population = std::move(next);
        for (const auto& ind : population) {
            if (ind.distance < individual.distance) {
                individual = ind;
            }
        }
    }

    const Individual& getBestIndividualStructure() const {
        return individual;
    }

    double getBestDistance() const { return individual.distance; }
    std::vector<int> getBestRoute() const { return individual.chromosome; }
};

int main(int argc, char* argv[]) {

    std::string filepath = "berlin52.tsp";
    int generations = 100;
    int popSize = 100;
    int tournamentSize = 50;
    unsigned int seed = 42;
    double knownOptimum = 7542.0;

    std::cout << "=================================================\n";
    std::cout << "🧬       C++ GENETIC ALGORITHM TSP SOLVER        \n";
    std::cout << "=================================================\n";

    std::vector<Point> nodes;
    try {
        nodes = parseTSPFile(filepath);
    } catch (const std::exception& e) {
        std::cerr << "❌ File Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "📂 Target Dataset : " << filepath << "\n";
    std::cout << "📍 Total Cities    : " << nodes.size() << "\n";
    std::cout << "👥 Population Size: " << popSize << "\n";
    std::cout << "⏳ Generations    : " << generations << "\n";
    std::cout << "-------------------------------------------------\n";
    std::cout << "🚀 Initiating evolution...\n\n";

    auto startTime = std::chrono::high_resolution_clock::now();
    GeneticAlgorithm ga(nodes, popSize, tournamentSize, seed);
    for (int gen = 1; gen <= generations; ++gen) {
        ga.evolveOneGeneration();
        if (gen % 50 == 0 || gen == generations) {
            std::cout << "\r🔄 Processing: [Gen " << std::setw(4) << gen << "/" << generations << "] "
                      << "Current Best Distance: " << std::fixed << std::setprecision(0)
                      << ga.getBestDistance() << " units" << std::flush;
        }
    }
    auto bestIndividual = ga.getBestIndividualStructure();
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    double accuracy = (knownOptimum / ga.getBestDistance()) * 100.0;

    std::cout << "\n\n=================================================\n";
    std::cout << "🎯 PERFORMANCE RESULTS\n";
    std::cout << "=================================================\n";
    std::cout << "⏱️  Total Execution Time: " << std::setprecision(4) << duration.count() << " seconds\n";
    std::cout << "🏆 Best Distance Found  : " << ga.getBestDistance() << " units\n";
    std::cout << "🏁 Target TSPLIB Optimum : " << knownOptimum << " units\n";
    std::cout << "📊 Accuracy : " << std::setprecision(2) << accuracy << "%\n";
    std::cout << "=================================================\n";
    return 0;
}
