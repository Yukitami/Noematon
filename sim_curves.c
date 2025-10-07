#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <GL/glut.h>
#include <omp.h>
#include <stdint.h> // For uint64_t
#include <unistd.h> // For getpid()
#include <ctype.h>
#include <stdbool.h> // Include OpenMP for parallelization

// --- NEW DEPENDENCIES ---
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- Configuration (Constants) ---
#define CANVAS_WIDTH 432
#define CANVAS_HEIGHT 324
#define MAX_SIM_STEPS 50000 // Changed from 50000 to transition to pattern writing sooner
#define MAX_PATTERN_WRITING_STEPS 50000 // Max steps for pattern writing phase
#define TARGET_IMAGE_FILENAME "IMG1.jpeg"
#define NAME_INTERVAL 400

// --- Limits for Naming & Rituals ---
#define MAX_NAMES 512
#define MAX_ACTIONS 16 // Max actions a worker can associate with a name
#define MAX_RITUALS 64
#define RITUAL_SUCCESS_THRESHOLD 30 // How many successful uses before a name can become a ritual
#define RITUAL_INVOCATION_INTERVAL 1000 // How often a ritual is invoked
#define RITUAL_DURATION 200 // How long a ritual is active
#define RITUAL_PLASTICITY_DURATION 50 // How long workers have increased plasticity during ritual
#define RITUAL_ENERGY_BONUS 25.0 // Energy bonus for workers during ritual
#define RITUAL_MAINTENANCE_INTERVAL 1000 // How often rituals are checked for fading/mutation
#define RITUAL_FADE_OUT_STEPS 20000 // Steps of disuse before a ritual fades out
#define RITUAL_DISSONANCE_THRESHOLD 500.0 // Accumulated dissonance to trigger ritual mutation/split
#define RITUAL_SPLIT_THRESHOLD 1000.0 // Higher dissonance to trigger ritual splitting
#define RITUAL_CONTEXT_AVG_ALPHA 0.995 // Alpha for moving average of ritual usage context

// --- Layer Configuration ---
#define CRITIC_COUNT 20
#define MANAGER_COUNT 40
#define WORKER_COUNT 60
#define CRITIC_LATENT_DIM 16
#define MANAGER_LATENT_DIM 16
#define WORKER_LATENT_DIM 16
#define CRITIC_TREE_DEPTH 8
#define MANAGER_TREE_DEPTH 10
#define WORKER_TREE_DEPTH 12
#define WORKER_NUM_CROSS_LINKS 100 // Number of random cross-links in worker program trees

// --- Critic Configuration ---
#define CRITIC_PHASE_INTERVAL 100 // How often critics evaluate the canvas
#define CRITIC_MDL_MAX_STEPS 10 // Not directly used in current critic implementation
#define CRITIC_UNEXPLAINED_STRUCTURE_THRESHOLD 0.1 // Not directly used in current critic implementation

// --- Economy & Learning (Constants) ---
#define INITIAL_ENERGY 100.0 // Starting energy for all machines
#define ENERGY_COST_PER_ACTION 0.1 // Energy consumed per machine action
#define WORKER_ENERGY_GAIN 50.0 // Multiplier for worker reward to energy conversion
#define WORKER_MANAGER_PAYMENT_RATIO 0.1 // Ratio of worker reward paid to manager
#define REWIRE_ENERGY_THRESHOLD 20.0 // Energy threshold below which a machine rewires
#define NUM_LINKS_TO_REWIRE 20 // Number of links to randomly rewire during rewiring
#define ERROR_PROPAGATION_DECAY 0.7 // Decay factor for error propagation up program tree
#define ERROR_VARIANCE_ALPHA 0.9 // Alpha for exponential moving average of error variance
#define HEBBIAN_THRESHOLD 0.001 // Minimum error magnitude for Hebbian learning to apply

// --- Taste Function Parameters ---
#define NOVELTY_ARCHIVE_SIZE 1000 // Number of past patterns to store for novelty search
#define PREDICTIVE_ERROR_OPTIMAL 0.05 // Optimal prediction error for "Goldilocks zone"
#define PREDICTIVE_ERROR_TOLERANCE 0.02 // How wide the "Goldilocks zone" is

// --- Self-Adaptation Tracking Constants ---
#define ADAPTATION_CHECK_INTERVAL 500 // How often system parameters are adapted
#define STAGNATION_THRESHOLD 1e-6 // Absolute threshold for MSE improvement (very small)
#define RELATIVE_STAGNATION_FACTOR 0.005 // 0.5% relative improvement considered non-stagnant
#define GOOD_ENOUGH_MSE 1e-4 // If MSE is below this, target is considered "mostly met"
#define MIN_TARGET_INFLUENCE 0.2 // Minimum target influence weight to prevent total abandonment
#define AESTHETIC_STAGNATION_THRESHOLD 1e-4 // For critic score improvement
#define PARAMETER_MUTATION_STRENGTH 0.1 // Magnitude of random mutation for parameters

// --- Forward Declarations for Data Structures ---
// These are defined here so the Simulation struct can reference them.
typedef struct ProgramTreeNode ProgramTreeNode;
typedef struct KernelElement KernelElement;
typedef struct WorkerKernel WorkerKernel;
typedef struct Machine Machine;
typedef struct NameAction NameAction;
typedef struct NameDictionary NameDictionary;
typedef struct NameStats NameStats;
typedef struct RitualModulator RitualModulator;
typedef struct NameRegistry NameRegistry;
typedef struct RitualRegistry RitualRegistry;
typedef struct SystemParams SystemParams;
typedef struct Simulation Simulation; // Declare Simulation struct

// --- Data Structures ---
// Represents a node in a machine's program tree (like a neuron or instruction)
struct ProgramTreeNode {
    char state_label; // A label for debugging/identification
    double *w_latent; // Weights for latent input vector
    int latent_dim; // Dimension of the latent vector
    double bias; // Bias term
    int move_x, move_y; // Spatial movement for agent's focus point
    ProgramTreeNode *next_left, *next_right; // Pointers for binary decision tree traversal
    ProgramTreeNode *parent; // Pointer to parent for error propagation
    ProgramTreeNode *cross_link; // Arbitrary link to another node for graph-like structure
    int target_x, target_y; // Current pixel coordinates this node is focused on
    double prediction_error; // Accumulated error for learning
    double error_variance; // Variance of error for adaptive learning rate
};

// Kernel element for directional actions
struct KernelElement {
    double angle;   // in radians
    double weight;  // strength of this direction
    double radius;  // step size
};

// Worker kernel structure
#define MAX_KERNEL_ACTIONS 32  // allow more than 8
struct WorkerKernel {
    KernelElement elements[MAX_KERNEL_ACTIONS];
    int count;
};

// Represents an agent (Critic, Manager, or Worker)
struct Machine {
    int id; // Unique ID for the machine
    ProgramTreeNode *root; // Root of its program tree
    ProgramTreeNode *current; // Current active node in the tree
    ProgramTreeNode **all_nodes; // Array of all nodes for easy access/rewiring
    int node_count; // Total number of nodes in the tree
    double energy; // Current energy level (Workers & Managers consume, Critics don't)
    WorkerKernel kernel; // Kernel for workers (replaced fixed 3x3)
    int plasticity_timer; // Timer for increased learning (e.g., during rituals)
    double local_lr_modifier; // Local learning rate multiplier
    double local_confidence_modifier; // Local confidence multiplier
    int worker_imprint_radius; // (Workers only) Multiplier for radius during rituals
};

// Defines an action associated with a name (worker-specific)
struct NameAction {
    char* name; // The name string
    double bias_mod; // Bias modification for worker's internal state
    double angle_mod; // Modification to angle
    double weight_mod; // Modification to weight
    double radius_mod; // Modification to radius
};

// Dictionary of actions a worker has learned for specific names
struct NameDictionary {
    NameAction actions[MAX_ACTIONS];
    int action_count;
};

// Statistics for a registered name
struct NameStats {
    char* name; // The name string (duplicated from registry)
    int success_count; // How many times this name led to success
    double total_reward; // Accumulated reward when this name was active
    int uses; // Total times this name was used
    double creation_context[WORKER_LATENT_DIM]; // Latent context when this name was created
};

// Modulator for a ritual
struct RitualModulator {
    int is_active; // Is this ritual currently active?
    double meaning_vector[WORKER_LATENT_DIM]; // The "meaning" of the ritual (latent vector)
    double goal_vector[WORKER_LATENT_DIM]; // The "goal" associated with the ritual (latent vector)
    long last_invoked_step; // Last simulation step this ritual was invoked
    double accumulated_dissonance; // How much the usage context deviates from meaning
    double current_usage_context_avg[WORKER_LATENT_DIM]; // Moving average of contexts during ritual use
    long usage_count; // How many times this ritual was used
};

// Centralized registry for all names
struct NameRegistry {
    char* registry[MAX_NAMES]; // Array of name strings
    int count; // Current number of names
    NameStats usage_stats[MAX_NAMES]; // Stats for each name
};

// Centralized registry and state for all rituals
struct RitualRegistry {
    char* registry[MAX_RITUALS]; // Array of ritual names
    int count; // Current number of rituals
    int active_idx; // Index of the currently active ritual (-1 if none)
    int active_timer; // Timer for active ritual duration
    RitualModulator modulators[MAX_RITUALS]; // Modulator data for each ritual
};

// System-wide Parameters Struct
struct SystemParams {
    double confidence_base;
    double confidence_gain;
    double reward_influence;
    double worker_base_lr;
    double worker_kernel_lr;
    double guidance_reward_weight;
    double ritual_goal_reward_weight;
    double ritual_plasticity_modifier;
    double critic_base_lr;
    double critic_target_weight;
    double critic_novelty_weight;
    double critic_repetition_penalty_weight;
    double critic_pattern_novelty_weight;
    double critic_pattern_repetition_penalty_weight;
    double taste_novelty_weight;
    double taste_predictive_weight;
    double target_influence_adapt_rate;
    double taste_influence_threshold_chaos;
    double random_influence_perturbation;
    int ritual_imprint_radius_increase;
};

// --- Main Simulation State Struct ---
// This struct holds all global state and parameters, passed to most functions.
struct Simulation {
    // Canvases
    double *global_canvas;
    double *target_canvas;
    double *guidance_latent_canvas;
    double *criticism_canvas;
    double *previous_global_canvas;
    double *pattern_canvas;
    double *previous_pattern_canvas;
    // Novelty Archive for Taste Function
    double *novelty_archive;
    int novelty_archive_ptr;
    // Simulation State
    int step; // Current simulation step (recreation phase)
    int pattern_step; // Current step in pattern writing phase
    int pattern_writing_mode; // Flag: 0 for recreation, 1 for pattern writing
    // Adaptive Influence Weight
    double target_influence_weight; // Weight for target influence on workers (adaptive)
    // Machine Arrays
    Machine critics[CRITIC_COUNT];
    Machine managers[MANAGER_COUNT];
    Machine workers[WORKER_COUNT];
    NameDictionary worker_name_dicts[WORKER_COUNT]; // Worker-specific name dictionaries
    // Registries
    NameRegistry name_registry;
    RitualRegistry ritual_registry;
    // System Parameters
    SystemParams params;
    // RNG State
    uint64_t xorshift_state;
    // Logging
    FILE *log_file;
    FILE *event_log_file; // New: for event logging
    // Adaptation Tracking
    double mse_at_window_start;
    double critic_score_at_window_start;
    double sum_prediction_error; // Accumulated prediction error for adaptive influence
    int taste_eval_count; // Count of taste evaluations for averaging
};

// --- Utility Function Prototypes (now take Simulation* as first arg) ---
// These prototypes must be above any function definitions that call them.
static uint64_t sim_xorshift(Simulation* sim);
static double sim_rand_double(Simulation* sim);
static int canvas_idx(int x, int y);
static void canvas_randomize(Simulation* sim, double *canvas);
static double canvas_evaluate_mse_loss(Simulation* sim);
static double node_evaluate_in_latent(ProgramTreeNode *node, double *latent);
static double latent_distance(double *v1, double *v2, int dim); // Euclidean distance
static double vector_dot_product(const double* v1, const double* v2, int dim); // New: Dot product
static double vector_magnitude(const double* v, int dim); // New: Vector magnitude
static void encode_context(const double *canvas, int cx, int cy, double *latent, int latent_dim);
static void encode_manager_context(Simulation* sim, int cx, int cy, double *latent, int latent_dim);
static void encode_critic_context(Simulation* sim, int cx, int cy, double *latent, int latent_dim, const double* current_canvas, const double* prev_canvas);
static ProgramTreeNode *node_new(Simulation* sim, char label, ProgramTreeNode *parent, int latent_dim);
static ProgramTreeNode *node_build_graph_like_tree(Simulation* sim, int depth, ProgramTreeNode *parent, Machine *mach, int *node_idx, int latent_dim);
static void node_propagate_error_up(ProgramTreeNode *node, double error);
static void machine_rewire(Simulation* sim, Machine *mach);
static void worker_init_kernel(Simulation* sim, Machine *mach);
static void save_canvas_snapshot(const char* filename, const double* canvas_to_save);
static void simulation_log_event(Simulation* sim, const char* message); // Prototype for logging function
static void draw_on_canvas(Simulation* sim, double *canvas, double *target_canvas, double target_influence, int x0, int y0, int x1, int y1, double intensity);

// Name Registry Module Prototypes
static void naming_init(Simulation* sim);
static char* naming_generate_from_latent(Simulation* sim, double* latent, int dim);
static void naming_add_to_global_pool(Simulation* sim, char* name, double* creation_context);
static void naming_interpret_for_manager(const char* name, int* out_x, int* out_y);
static void naming_assign_to_worker(Simulation* sim, int worker_id, const char* name);
static void naming_phase(Simulation* sim);
static void interpretation_phase(Simulation* sim);

// Ritual Registry Module Prototypes
static void ritual_init(Simulation* sim);
static void ritual_remove(Simulation* sim, int ritual_idx);
static void ritual_promote_name(Simulation* sim, int name_idx);
static void ritual_invocation_phase(Simulation* sim);
static void ritual_maintenance_phase(Simulation* sim);

// Critic Module Prototypes
static void critic_init(Simulation* sim, Machine *critic_mach, int id);
static void critic_phase(Simulation* sim);

// Manager Module Prototypes
static void manager_init(Simulation* sim, Machine *manager_mach, int id);
static void manager_step(Simulation* sim, Machine *mach);

// Worker Module Prototypes
static void worker_init(Simulation* sim, Machine *worker_mach, int id);
static void worker_step(Simulation* sim, Machine *mach);

// Taste Function Prototypes
static void novelty_archive_init(Simulation* sim);
static double novelty_archive_get_nn_distance(Simulation* sim, double *query_vector, int dim);
static void novelty_archive_add(Simulation* sim, double *vector, int dim);
static double worker_calculate_taste(Simulation* sim, int cx, int cy, double *worker_latent_context);

// Simulation Core Logic Prototypes
static void simulation_adapt_parameters(Simulation* sim);
static void simulation_step(Simulation* sim);
static void simulation_pattern_writing_step(Simulation* sim);
static void simulation_cleanup(Simulation* sim); // Cleanup prototype
static int load_target_image(Simulation* sim);
static void init_default_params(Simulation* sim);
static Simulation* simulation_init(void); // Main initialization function

// GLUT callback wrappers (need static global pointer)
static Simulation* g_sim_instance = NULL; // Global pointer for GLUT callbacks
static void display_canvas(void);
static void update_simulation(void);
void glut_exit_handler(void); // Global function for atexit

// --- Utility Functions Implementations ---
// Random number generation
static uint64_t sim_xorshift(Simulation* sim) {
    sim->xorshift_state ^= sim->xorshift_state >> 12;
    sim->xorshift_state ^= sim->xorshift_state << 25;
    sim->xorshift_state ^= sim->xorshift_state >> 27;
    return sim->xorshift_state * 2685821657736338717ULL;
}

static double sim_rand_double(Simulation* sim) { // stochastic step
    return (double)(sim_xorshift(sim) & 0xFFFFFF) / 16777215.0;
}

// Canvas indexing with wrap-around
static int canvas_idx(int x, int y) {
    x = (x % CANVAS_WIDTH + CANVAS_WIDTH) % CANVAS_WIDTH;
    y = (y % CANVAS_HEIGHT + CANVAS_HEIGHT) % CANVAS_HEIGHT;
    return y * CANVAS_WIDTH + x;
}

// Canvas operations
static void canvas_randomize(Simulation* sim, double *canvas) {
#pragma omp parallel for
    for (int i = 0; i < CANVAS_WIDTH * CANVAS_HEIGHT; i++) {
        // Note: Using sim_xorshift in parallel loop requires careful handling of RNG state
        // For simplicity here, each thread might get a slightly different sequence if not explicitly seeded per thread.
        // For truly deterministic parallel RNG, a more complex parallel RNG library is needed.
        // For this refactor, we accept potential non-determinism across runs if threads grab state non-sequentially.
        canvas[i] = (double)(sim_xorshift(sim) & 0xFFFFFF) / 16777215.0; // stochastic step
    }
}

static double canvas_evaluate_mse_loss(Simulation* sim) {
    double total_loss = 0.0;
#pragma omp parallel for reduction(+:total_loss)
    for (int i = 0; i < CANVAS_WIDTH * CANVAS_HEIGHT; i++) {
        double diff = sim->global_canvas[i] - sim->target_canvas[i];
        total_loss += diff * diff;
    }
    return total_loss / (CANVAS_WIDTH * CANVAS_HEIGHT);
}

// ProgramTreeNode evaluation
static double node_evaluate_in_latent(ProgramTreeNode *node, double *latent) {
    double sum = node->bias;
    for (int i = 0; i < node->latent_dim; i++) {
        sum += node->w_latent[i] * latent[i];
    }
    return tanh(sum);
}

// Latent vector Euclidean distance
static double latent_distance(double *v1, double *v2, int dim) {
    double dist = 0.0;
    for(int i=0; i<dim; i++) {
        double d = v1[i] - v2[i];
        dist += d * d;
    }
    return sqrt(dist);
}

// New: Dot product for cosine similarity
static double vector_dot_product(const double* v1, const double* v2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

// New: Vector magnitude for cosine similarity
static double vector_magnitude(const double* v, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

// Worker context encoding (canvas only)
static void encode_context(const double *canvas, int cx, int cy, double *latent, int latent_dim) {
    double context[9];
    int i = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            context[i] = canvas[canvas_idx(cx + x, cy + y)];
            i++;
        }
    }
    for (int d = 0; d < latent_dim; d++) {
        latent[d] = 0.0;
        for (int j = 0; j < 9; j++) {
            latent[d] += sin(0.5 * d * j) * context[j];
        }
        latent[d] = tanh(latent[d]);
    }
}

// Manager context encoding (Global_Canvas & Criticism_Canvas)
static void encode_manager_context(Simulation* sim, int cx, int cy, double *latent, int latent_dim) {
    double context_global[9], context_critic[9];
    int i = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            context_global[i] = sim->global_canvas[canvas_idx(cx + x, cy + y)];
            context_critic[i] = sim->criticism_canvas[canvas_idx(cx + x, cy + y)];
            i++;
        }
    }
    for (int d = 0; d < latent_dim; d++) {
        latent[d] = 0.0;
        for (int j = 0; j < 9; j++) {
            latent[d] += sin(0.5 * d * j) * context_global[j];
            latent[d] += cos(0.5 * d * j) * context_critic[j];
        }
        latent[d] = tanh(latent[d]);
    }
}

static void encode_critic_context(Simulation* sim, int cx, int cy, double *latent, int latent_dim, const double* current_canvas, const double* prev_canvas) {
    double context_current[9], context_prev[9];
    int i = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            int current_idx = canvas_idx(cx + x, cy + y);
            context_current[i] = current_canvas[current_idx];
            context_prev[i] = prev_canvas[current_idx];
            i++;
        }
    }
    for (int d = 0; d < latent_dim; d++) {
        latent[d] = 0.0;
        for (int j = 0; j < 9; j++) {
            latent[d] += sin(0.5 * d * j) * context_current[j];
            latent[d] += cos(0.5 * d * j) * context_prev[j];
        }
        latent[d] = tanh(latent[d]);
    }
}

// ProgramTreeNode operations
static ProgramTreeNode *node_new(Simulation* sim, char label, ProgramTreeNode *parent, int latent_dim) {
    ProgramTreeNode *node = (ProgramTreeNode*)malloc(sizeof(ProgramTreeNode));
    if (!node) {
        perror("Failed to allocate ProgramTreeNode");
        exit(EXIT_FAILURE);
    }
    node->state_label = label;
    node->parent = parent;
    node->latent_dim = latent_dim;
    node->w_latent = (double*)malloc(latent_dim * sizeof(double));
    if (!node->w_latent) {
        perror("Failed to allocate w_latent");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < latent_dim; i++) node->w_latent[i] = (sim_rand_double(sim) * 2.0 - 1.0); // stochastic step
    node->bias = (sim_rand_double(sim) * 2.0 - 1.0); // stochastic step
    node->move_x = (int)(sim_xorshift(sim) % 5) - 2; // stochastic step
    node->move_y = (int)(sim_xorshift(sim) % 5) - 2; // stochastic step
    node->target_x = sim_xorshift(sim) % CANVAS_WIDTH; // stochastic step
    node->target_y = sim_xorshift(sim) % CANVAS_HEIGHT; // stochastic step
    node->next_left = node->next_right = node->cross_link = NULL;
    node->prediction_error = 0.0;
    node->error_variance = 1.0;
    return node;
}

static ProgramTreeNode *node_build_graph_like_tree(Simulation* sim, int depth, ProgramTreeNode *parent, Machine *mach, int *node_idx, int latent_dim) {
    char label = 'A' + (sim_xorshift(sim) % 26); // stochastic step
    ProgramTreeNode *node = node_new(sim, label, parent, latent_dim);
    mach->all_nodes[(*node_idx)++] = node;
    if (depth <= 0) {
        node->next_left = node;
        node->next_right = node;
    } else {
        node->next_left = node_build_graph_like_tree(sim, depth - 1, node, mach, node_idx, latent_dim);
        node->next_right = node_build_graph_like_tree(sim, depth - 1, node, mach, node_idx, latent_dim);
    }
    return node;
}

static void node_propagate_error_up(ProgramTreeNode *node, double error) {
    while (node) {
        node->prediction_error += error;
        error *= ERROR_PROPAGATION_DECAY;
        node = node->parent;
    }
}

// Machine operations
static void machine_rewire(Simulation* sim, Machine *mach) {
    for (int i = 0; i < NUM_LINKS_TO_REWIRE; i++) {
        int a = sim_xorshift(sim) % mach->node_count; // stochastic step
        int b = sim_xorshift(sim) % mach->node_count; // stochastic step
        if (a != b) mach->all_nodes[a]->cross_link = mach->all_nodes[b];
    }
    mach->energy = REWIRE_ENERGY_THRESHOLD + 10.0;
    mach->local_lr_modifier *= (1.0 + (sim_rand_double(sim) - 0.5) * 0.1); // stochastic step
    mach->local_confidence_modifier *= (1.0 + (sim_rand_double(sim) - 0.5) * 0.1); // stochastic step
    if (mach->local_lr_modifier < 0.5) mach->local_lr_modifier = 0.5;
    if (mach->local_lr_modifier > 1.5) mach->local_lr_modifier = 1.5;
    if (mach->local_confidence_modifier < 0.5) mach->local_confidence_modifier = 0.5;
    if (mach->local_confidence_modifier > 1.5) mach->local_confidence_modifier = 1.5;
    simulation_log_event(sim, "Machine rewiring triggered.");
}

static void worker_init_kernel(Simulation* sim, Machine *mach) {
    mach->kernel.count = MAX_KERNEL_ACTIONS;
    for (int i = 0; i < mach->kernel.count; i++) {
        mach->kernel.elements[i].angle = sim_rand_double(sim) * 2.0 * M_PI; // random angle 0–2π
        mach->kernel.elements[i].radius = 1.0 + floor(sim_rand_double(sim) * 3.0); // step length 1–3
        mach->kernel.elements[i].weight = sim_rand_double(sim); // weight for ritual reinforcement
    }
}

// --- Name Registry Module (`naming_`) ---
static void naming_init(Simulation* sim) {
    sim->name_registry.count = 0;
    for (int i = 0; i < MAX_NAMES; ++i) {
        sim->name_registry.registry[i] = NULL;
        sim->name_registry.usage_stats[i].name = NULL;
    }
}

static char* naming_generate_from_latent(Simulation* sim, double* latent, int dim) {
    static char buffer[16];
    unsigned int hash = 0;
    for (int i = 0; i < dim; ++i) {
        hash = (hash * 16777619) ^ (unsigned int)(fabs(latent[i]) * 1000.0);
    }
    snprintf(buffer, sizeof(buffer), "n%04X", hash & 0xFFFF);
    return strdup(buffer);
}

static void naming_add_to_global_pool(Simulation* sim, char* name, double* creation_context) {
    int target_idx = -1;
#pragma omp critical // Protect global name registry modification
    {
        if (sim->name_registry.count >= MAX_NAMES) {
            // Simple eviction: replace a random name from the first half
            target_idx = sim_xorshift(sim) % (MAX_NAMES / 2); // stochastic step
            if (sim->name_registry.registry[target_idx]) free(sim->name_registry.registry[target_idx]);
            if (sim->name_registry.usage_stats[target_idx].name) free(sim->name_registry.usage_stats[target_idx].name);
        } else {
            target_idx = sim->name_registry.count++;
        }
        sim->name_registry.registry[target_idx] = name; // Takes ownership of the dynamically allocated name
        sim->name_registry.usage_stats[target_idx].name = strdup(name); // Duplicate for stats struct
        sim->name_registry.usage_stats[target_idx].success_count = 0;
        sim->name_registry.usage_stats[target_idx].total_reward = 0;
        sim->name_registry.usage_stats[target_idx].uses = 0;
        if (creation_context) {
            memcpy(sim->name_registry.usage_stats[target_idx].creation_context, creation_context, sizeof(double) * WORKER_LATENT_DIM);
        } else {
            memset(sim->name_registry.usage_stats[target_idx].creation_context, 0, sizeof(double) * WORKER_LATENT_DIM);
        }
    }
    simulation_log_event(sim, "Name created.");
}

static void naming_interpret_for_manager(const char* name, int* out_x, int* out_y) {
    unsigned int h = 0;
    for (const char* p = name; *p; ++p) h = (h * 1315423911) ^ (*p << 5) ^ (*p >> 2);
    *out_x = h % CANVAS_WIDTH;
    *out_y = (h / CANVAS_WIDTH) % CANVAS_HEIGHT;
}

static void naming_assign_to_worker(Simulation* sim, int worker_id, const char* name) {
    if (strchr(name, '_')) return; // Don't assign composite names directly to workers for action interpretation
    NameDictionary *dict = &sim->worker_name_dicts[worker_id];
    if (dict->action_count >= MAX_ACTIONS) return;
    // Check if name already exists to avoid duplicates
    for(int i=0; i<dict->action_count; ++i) {
        if(strcmp(dict->actions[i].name, name) == 0) {
            return;
        }
    }
    NameAction *a = &dict->actions[dict->action_count++];
    a->name = strdup(name);
    a->bias_mod = (sim_rand_double(sim) * 0.1 - 0.05); // stochastic step
    a->angle_mod = (sim_rand_double(sim) * M_PI / 4 - M_PI / 8); // stochastic step
    a->weight_mod = (sim_rand_double(sim) - 0.5) * 0.1; // stochastic step
    a->radius_mod = (sim_rand_double(sim) - 0.5) * 1.0; // stochastic step
}

static void naming_phase(Simulation* sim) {
    if (sim->step > 0 && sim->step % NAME_INTERVAL == 0) {
        char* new_name = NULL;
        double creation_context[WORKER_LATENT_DIM] = {0};
        // Option 1: Generate a completely new name based on a random manager's context
        if (sim->name_registry.count < 10 || (sim_xorshift(sim) % 10) < 3) { // stochastic step
            int m = sim_xorshift(sim) % MANAGER_COUNT; // stochastic step
            ProgramTreeNode *cur = sim->managers[m].current;
            if (!cur) return;
            encode_context(sim->global_canvas, cur->target_x, cur->target_y, creation_context, WORKER_LATENT_DIM);
            new_name = naming_generate_from_latent(sim, creation_context, WORKER_LATENT_DIM);
        }
        // Option 2: Combine two existing names to create a composite name
        else if (sim->name_registry.count >= 2) {
            int idx1 = sim_xorshift(sim) % sim->name_registry.count; // stochastic step
            int idx2 = sim_xorshift(sim) % sim->name_registry.count; // stochastic step
            if (idx1 == idx2) idx2 = (idx2 + 1) % sim->name_registry.count;
            char* name1 = sim->name_registry.registry[idx1];
            char* name2 = sim->name_registry.registry[idx2];
            // Prevent combining composite names to avoid excessively long names
            if (strchr(name1, '_') || strchr(name2, '_')) return;
            new_name = (char*)malloc(strlen(name1) + strlen(name2) + 2); // +1 for '_', +1 for '\0'
            if (new_name) {
                sprintf(new_name, "%s_%s", name1, name2);
                // Average the creation contexts for the new composite name
                for(int i=0; i<WORKER_LATENT_DIM; i++) {
                    creation_context[i] = (sim->name_registry.usage_stats[idx1].creation_context[i] + sim->name_registry.usage_stats[idx2].creation_context[i]) / 2.0;
                }
            }
        }
        if (new_name) {
            naming_add_to_global_pool(sim, new_name, creation_context);
        }
    }
}

static void interpretation_phase(Simulation* sim) {
    if (sim->name_registry.count == 0) return;
    // Managers interpret the latest created name
    const char* latest_name = sim->name_registry.registry[sim->name_registry.count - 1];
    for (int m = 0; m < MANAGER_COUNT; m++) {
        if (!sim->managers[m].current) continue;
        int tx, ty;
        naming_interpret_for_manager(latest_name, &tx, &ty);
        sim->managers[m].current->target_x = tx;
        sim->managers[m].current->target_y = ty;
    }
    // Workers randomly learn existing names
    for (int w = 0; w < WORKER_COUNT; w++) {
        if ((sim_xorshift(sim) & 3) == 0) { // stochastic step
            // Randomly assign a name to some workers
            const char* name_to_learn = sim->name_registry.registry[sim_xorshift(sim) % sim->name_registry.count]; // stochastic step
            naming_assign_to_worker(sim, w, name_to_learn);
        }
    }
}

// --- Ritual Registry Module (`ritual_`) ---
static void ritual_init(Simulation* sim) {
    sim->ritual_registry.count = 0;
    sim->ritual_registry.active_idx = -1;
    sim->ritual_registry.active_timer = 0;
    for (int i = 0; i < MAX_RITUALS; ++i) {
        sim->ritual_registry.registry[i] = NULL;
        sim->ritual_registry.modulators[i].is_active = 0;
    }
}

static void ritual_remove(Simulation* sim, int ritual_idx) {
    if (ritual_idx < 0 || ritual_idx >= sim->ritual_registry.count) return;
#pragma omp critical
    {
        if (ritual_idx < sim->ritual_registry.count) { // Double check in critical section
            char log_msg[128];
            sprintf(log_msg, "Ritual '%s' faded out due to disuse.", sim->ritual_registry.registry[ritual_idx]);
            simulation_log_event(sim, log_msg);
            free(sim->ritual_registry.registry[ritual_idx]); // Free memory for the name
            // Shift remaining rituals to fill the gap
            for (int i = ritual_idx; i < sim->ritual_registry.count - 1; i++) {
                sim->ritual_registry.registry[i] = sim->ritual_registry.registry[i + 1];
                memcpy(&sim->ritual_registry.modulators[i], &sim->ritual_registry.modulators[i + 1], sizeof(RitualModulator));
            }
            sim->ritual_registry.count--; // Decrement ritual count
            if (sim->ritual_registry.active_idx == ritual_idx) {
                // If the removed ritual was active
                sim->ritual_registry.active_idx = -1; // Deactivate it
                sim->ritual_registry.active_timer = 0;
                // Reset worker imprint radius if the active ritual was removed
#pragma omp parallel for
                for(int w=0; w<WORKER_COUNT; w++) {
                    sim->workers[w].worker_imprint_radius = 1;
                }
            } else if (sim->ritual_registry.active_idx > ritual_idx) {
                // If active ritual shifted
                sim->ritual_registry.active_idx--;
            }
        }
    }
}

static void ritual_promote_name(Simulation* sim, int name_idx) {
    if (sim->name_registry.usage_stats[name_idx].success_count > RITUAL_SUCCESS_THRESHOLD) {
#pragma omp critical
        {
            if (sim->name_registry.usage_stats[name_idx].success_count > RITUAL_SUCCESS_THRESHOLD) { // Double check
                if (sim->ritual_registry.count < MAX_RITUALS) {
                    int already_ritual = 0;
                    for(int j=0; j<sim->ritual_registry.count; ++j) {
                        if(strcmp(sim->ritual_registry.registry[j], sim->name_registry.usage_stats[name_idx].name) == 0) {
                            already_ritual=1;
                            break;
                        }
                    }
                    if(!already_ritual){ // Only add if not already a ritual
                        int r_idx = sim->ritual_registry.count++;
                        sim->ritual_registry.registry[r_idx] = strdup(sim->name_registry.usage_stats[name_idx].name);
                        sim->ritual_registry.modulators[r_idx].is_active = 1;
                        // Ritual meaning and goal are initialized from the name's creation context
                        memcpy(sim->ritual_registry.modulators[r_idx].meaning_vector, sim->name_registry.usage_stats[name_idx].creation_context, sizeof(double) * WORKER_LATENT_DIM);
                        memcpy(sim->ritual_registry.modulators[r_idx].goal_vector, sim->name_registry.usage_stats[name_idx].creation_context, sizeof(double) * WORKER_LATENT_DIM);
                        memcpy(sim->ritual_registry.modulators[r_idx].current_usage_context_avg, sim->name_registry.usage_stats[name_idx].creation_context, sizeof(double) * WORKER_LATENT_DIM);
                        sim->ritual_registry.modulators[r_idx].last_invoked_step = sim->step;
                        sim->ritual_registry.modulators[r_idx].accumulated_dissonance = 0.0;
                        sim->ritual_registry.modulators[r_idx].usage_count = 0;
                        char log_msg[128];
                        sprintf(log_msg, "Ritualized: %s", sim->ritual_registry.registry[r_idx]);
                        simulation_log_event(sim, log_msg);
                    }
                }
            }
            sim->name_registry.usage_stats[name_idx].success_count = 0; // Reset success count after ritualization
        }
    }
}

static void ritual_invocation_phase(Simulation* sim) {
    if (sim->ritual_registry.active_timer > 0) {
        sim->ritual_registry.active_timer--;
        if (sim->ritual_registry.active_timer == 0) {
            sim->ritual_registry.active_idx = -1; // End active ritual
            // Reset worker imprint radius when ritual ends
#pragma omp parallel for
            for(int w=0; w<WORKER_COUNT; w++) {
                sim->workers[w].worker_imprint_radius = 1; // Reset to default 3x3
            }
        }
    } else if (sim->ritual_registry.count > 0 && sim->step % RITUAL_INVOCATION_INTERVAL == 0) {
        sim->ritual_registry.active_idx = sim_xorshift(sim) % sim->ritual_registry.count; // stochastic step
        // Select a random ritual toactivate
        sim->ritual_registry.active_timer = RITUAL_DURATION; // Set ritual duration
        sim->ritual_registry.modulators[sim->ritual_registry.active_idx].last_invoked_step = sim->step;
        // Activate plasticity for workers during ritual
#pragma omp parallel for
        for(int w=0; w<WORKER_COUNT; w++) {
            sim->workers[w].plasticity_timer = RITUAL_PLASTICITY_DURATION;
            // Increase worker imprint radius during ritual
            sim->workers[w].worker_imprint_radius = 1 + sim->params.ritual_imprint_radius_increase;
        }
        char log_msg[128];
        sprintf(log_msg, "Invoking Ritual: %s", sim->ritual_registry.registry[sim->ritual_registry.active_idx]);
        simulation_log_event(sim, log_msg);
    }
}

static void ritual_maintenance_phase(Simulation* sim) {
    if (sim->step == 0 || sim->step % RITUAL_MAINTENANCE_INTERVAL != 0) return;
    for (int i = 0; i < sim->ritual_registry.count; i++) {
        // Fade out rituals due to disuse
        if (sim->step - sim->ritual_registry.modulators[i].last_invoked_step > RITUAL_FADE_OUT_STEPS) {
            ritual_remove(sim, i);
            i--; // Decrement i because the array shifted
            continue;
        }
        // Check for dissonance and mutate/split rituals
        if (sim->ritual_registry.modulators[i].accumulated_dissonance > RITUAL_DISSONANCE_THRESHOLD) {
#pragma omp critical
            {
                if (sim->ritual_registry.modulators[i].accumulated_dissonance > RITUAL_DISSONANCE_THRESHOLD) { // Double check
                    if (sim->ritual_registry.modulators[i].accumulated_dissonance > RITUAL_SPLIT_THRESHOLD && sim->ritual_registry.count < MAX_RITUALS) {
                        char log_msg[128];
                        sprintf(log_msg, "Ritual '%s' is splitting due to high dissonance!", sim->ritual_registry.registry[i]);
                        simulation_log_event(sim, log_msg);
                        char* old_name = sim->ritual_registry.registry[i];
                        char* new_name_str = (char*)malloc(strlen(old_name) + 5); // "_v2\0"
                        if (new_name_str) sprintf(new_name_str, "%s_v2", old_name);
                        int r_idx = sim->ritual_registry.count++;
                        sim->ritual_registry.registry[r_idx] = new_name_str;
                        sim->ritual_registry.modulators[r_idx].is_active = 1;
                        // New ritual's meaning/goal based on current usage context
                        memcpy(sim->ritual_registry.modulators[r_idx].meaning_vector, sim->ritual_registry.modulators[i].current_usage_context_avg, sizeof(double) * WORKER_LATENT_DIM);
                        memcpy(sim->ritual_registry.modulators[r_idx].goal_vector, sim->ritual_registry.modulators[i].current_usage_context_avg, sizeof(double) * WORKER_LATENT_DIM);
                        memcpy(sim->ritual_registry.modulators[r_idx].current_usage_context_avg, sim->ritual_registry.modulators[i].current_usage_context_avg, sizeof(double) * WORKER_LATENT_DIM);
                        sim->ritual_registry.modulators[r_idx].last_invoked_step = sim->step;
                        sim->ritual_registry.modulators[r_idx].accumulated_dissonance = 0.0;
                        sim->ritual_registry.modulators[r_idx].usage_count = 0;
                    } else {
                        char log_msg[128];
                        sprintf(log_msg, "Ritual '%s' is mutating due to dissonance.", sim->ritual_registry.registry[i]);
                        simulation_log_event(sim, log_msg);
                        // Mutate ritual's meaning/goal if dissonance is moderate
                        memcpy(sim->ritual_registry.modulators[i].meaning_vector, sim->ritual_registry.modulators[i].current_usage_context_avg, sizeof(double) * WORKER_LATENT_DIM);
                        memcpy(sim->ritual_registry.modulators[i].goal_vector, sim->ritual_registry.modulators[i].current_usage_context_avg, sizeof(double) * WORKER_LATENT_DIM);
                    }
                    sim->ritual_registry.modulators[i].accumulated_dissonance = 0.0; // Reset dissonance
                }
            }
        }
    }
}

// --- Critic Module (`critic_`) ---
static void critic_init(Simulation* sim, Machine *critic_mach, int id) {
    critic_mach->id = id;
    critic_mach->node_count = (1 << (CRITIC_TREE_DEPTH + 1)) - 1;
    critic_mach->all_nodes = (ProgramTreeNode**)malloc(critic_mach->node_count * sizeof(ProgramTreeNode*));
    if (!critic_mach->all_nodes) {
        perror("Failed to allocate critic nodes");
        exit(EXIT_FAILURE);
    }
    int ni=0;
    critic_mach->root = node_build_graph_like_tree(sim, CRITIC_TREE_DEPTH, NULL, critic_mach, &ni, CRITIC_LATENT_DIM);
    critic_mach->current = critic_mach->root;
    critic_mach->energy = INITIAL_ENERGY; // Critics don't use energy in this model, but initialized
}

static void critic_phase(Simulation* sim) {
    // Only run critic phase at intervals, or during pattern writing at a different interval
    int current_phase_step = sim->pattern_writing_mode ? sim->pattern_step : sim->step;
    if (current_phase_step % CRITIC_PHASE_INTERVAL != 0) return;
#pragma omp parallel for
    for (int c = 0; c < CRITIC_COUNT; c++) {
        Machine *mach = &sim->critics[c];
        if (!mach->current) continue;
        mach->current->target_x = sim_xorshift(sim) % CANVAS_WIDTH; // stochastic step
        mach->current->target_y = sim_xorshift(sim) % CANVAS_HEIGHT; // stochastic step
        int cx = mach->current->target_x;
        int cy = mach->current->target_y;
        cx = (cx % CANVAS_WIDTH + CANVAS_WIDTH) % CANVAS_WIDTH;
        cy = (cy % CANVAS_HEIGHT + CANVAS_HEIGHT) % CANVAS_HEIGHT;
        double latent_in[CRITIC_LATENT_DIM];
        double critic_raw_output;
        double ground_truth_score;
        if (!sim->pattern_writing_mode) {
            // Recreation Phase Critic Logic (judging Global_Canvas)
            encode_critic_context(sim, cx, cy, latent_in, CRITIC_LATENT_DIM, sim->global_canvas, sim->previous_global_canvas);
            critic_raw_output = node_evaluate_in_latent(mach->current, latent_in);
            double critic_output_score = (critic_raw_output + 1.0) / 2.0;
            double diff_target = sim->global_canvas[canvas_idx(cx, cy)] - sim->target_canvas[canvas_idx(cx, cy)];
            double diff_prev = sim->global_canvas[canvas_idx(cx, cy)] - sim->previous_global_canvas[canvas_idx(cx, cy)];
            double target_match_score = 1.0 - (diff_target * diff_target);
            double novelty_score = fabs(diff_prev);
            double repetition_score = 1.0 - fabs(diff_prev);
            ground_truth_score = (target_match_score * sim->params.critic_target_weight) +
                                 (novelty_score * sim->params.critic_novelty_weight) -
                                 (repetition_score * sim->params.critic_repetition_penalty_weight);
            ground_truth_score = fmax(0.0, fmin(1.0, ground_truth_score));
            sim->criticism_canvas[canvas_idx(cx, cy)] = critic_output_score; // Update criticism canvas
        } else {
            // Pattern Writing Phase Critic Logic (judging Pattern_Canvas)
            encode_critic_context(sim, cx, cy, latent_in, CRITIC_LATENT_DIM, sim->pattern_canvas, sim->previous_pattern_canvas);
            critic_raw_output = node_evaluate_in_latent(mach->current, latent_in);
            double critic_output_score = (critic_raw_output + 1.0) / 2.0;
            double diff_prev_pattern = sim->pattern_canvas[canvas_idx(cx, cy)] - sim->previous_pattern_canvas[canvas_idx(cx, cy)];
            double novelty_score_pattern = fabs(diff_prev_pattern);
            double repetition_score_pattern = 1.0 - fabs(diff_prev_pattern);
            // Critics during pattern writing only judge novelty and repetition of patterns
            ground_truth_score = (novelty_score_pattern * sim->params.critic_pattern_novelty_weight) -
                                 (repetition_score_pattern * sim->params.critic_pattern_repetition_penalty_weight);
            ground_truth_score = fmax(0.0, fmin(1.0, ground_truth_score));
            sim->criticism_canvas[canvas_idx(cx, cy)] = critic_output_score; // Still update criticism canvas
        }
        double critic_prediction_error = ground_truth_score - ((critic_raw_output + 1.0) / 2.0);
        node_propagate_error_up(mach->current, critic_prediction_error);
        ProgramTreeNode *node = mach->current;
        while(node != NULL) {
            if (fabs(node->prediction_error) > HEBBIAN_THRESHOLD) {
                double e = node->prediction_error;
                node->error_variance = (ERROR_VARIANCE_ALPHA * node->error_variance) + ((1.0 - ERROR_VARIANCE_ALPHA) * e * e);
                double learning_rate = sim->params.critic_base_lr;
                for (int d = 0; d < CRITIC_LATENT_DIM; d++) {
                    node->w_latent[d] += learning_rate * (e * latent_in[d] / (node->error_variance + latent_in[d]*latent_in[d]));
                }
                node->bias += learning_rate * (e / (node->error_variance + 1.0));
            }
            node->prediction_error = 0.0;
            node = node->parent;
        }
        mach->current = (critic_raw_output < 0) ? mach->current->next_left : mach->current->next_right;
        if (!mach->current) mach->current = mach->root;
    }
}

// --- Manager Module (`manager_`) ---
static void manager_init(Simulation* sim, Machine *manager_mach, int id) {
    manager_mach->id = id;
    manager_mach->node_count = (1 << (MANAGER_TREE_DEPTH + 1)) - 1;
    manager_mach->all_nodes = (ProgramTreeNode**)malloc(manager_mach->node_count * sizeof(ProgramTreeNode*));
    if (!manager_mach->all_nodes) {
        perror("Failed to allocate manager nodes");
        exit(EXIT_FAILURE);
    }
    int ni=0;
    manager_mach->root = node_build_graph_like_tree(sim, MANAGER_TREE_DEPTH, NULL, manager_mach, &ni, MANAGER_LATENT_DIM);
    manager_mach->current = manager_mach->root;
    manager_mach->energy = INITIAL_ENERGY;
}

static void manager_step(Simulation* sim, Machine *mach) {
    if (!mach->current || mach->energy <= 0) return;
    mach->energy -= ENERGY_COST_PER_ACTION;
    int cx = mach->current->target_x;
    int cy = mach->current->target_y;
    double latent_in[MANAGER_LATENT_DIM];
    encode_manager_context(sim, cx, cy, latent_in, MANAGER_LATENT_DIM);
    double manager_output[MANAGER_LATENT_DIM];
    for(int d=0; d<MANAGER_LATENT_DIM; d++) {
        manager_output[d] = node_evaluate_in_latent(mach->current, latent_in);
    }
    memcpy(&sim->guidance_latent_canvas[canvas_idx(cx, cy) * MANAGER_LATENT_DIM], manager_output, sizeof(double) * MANAGER_LATENT_DIM);
    mach->current = (manager_output[0] < 0) ? mach->current->next_left : mach->current->next_right;
    if (mach->current) {
        mach->current->target_x = cx + mach->current->move_x;
        mach->current->target_y = cy + mach->current->move_y;
        mach->current->target_x = (mach->current->target_x % CANVAS_WIDTH + CANVAS_WIDTH) % CANVAS_WIDTH;
        mach->current->target_y = (mach->current->target_y % CANVAS_HEIGHT + CANVAS_HEIGHT) % CANVAS_HEIGHT;
    } else {
        mach->current = mach->root;
    }
}

// --- Worker Module (`worker_`) ---
static void worker_init(Simulation* sim, Machine *worker_mach, int id) {
    worker_mach->id = id;
    worker_mach->node_count = (1 << (WORKER_TREE_DEPTH + 1)) - 1;
    worker_mach->all_nodes = (ProgramTreeNode**)malloc(worker_mach->node_count * sizeof(ProgramTreeNode*));
    if (!worker_mach->all_nodes) {
        perror("Failed to allocate worker nodes");
        exit(EXIT_FAILURE);
    }
    int ni=0;
    worker_mach->root = node_build_graph_like_tree(sim, WORKER_TREE_DEPTH, NULL, worker_mach, &ni, WORKER_LATENT_DIM);
    worker_mach->current = worker_mach->root;
    worker_mach->energy = INITIAL_ENERGY;
    worker_init_kernel(sim, worker_mach);
    sim->worker_name_dicts[id].action_count = 0;
    worker_mach->plasticity_timer = 0;
    worker_mach->local_lr_modifier = 1.0;
    worker_mach->local_confidence_modifier = 1.0;
    worker_mach->worker_imprint_radius = 1; // Initialize default imprint radius
    for(int j=0; j<WORKER_NUM_CROSS_LINKS; j++) { // Add cross-links for graph-like structure
        int a = sim_xorshift(sim) % worker_mach->node_count; // stochastic step
        int b = sim_xorshift(sim) % worker_mach->node_count; // stochastic step
        if (a != b) worker_mach->all_nodes[a]->cross_link = worker_mach->all_nodes[b];
    }
}

// Taste Function Implementations
static void novelty_archive_init(Simulation* sim) {
    // Initialize archive with zeros
    for (int i = 0; i < NOVELTY_ARCHIVE_SIZE * WORKER_LATENT_DIM; ++i) {
        sim->novelty_archive[i] = 0.0;
    }
    sim->novelty_archive_ptr = 0;
}

// Modified to use Cosine Distance (1 - Cosine Similarity)
static double novelty_archive_get_nn_distance(Simulation* sim, double *query_vector, int dim) {
    double query_magnitude = vector_magnitude(query_vector, dim);
    if (query_magnitude < 1e-9) return 1.0; // Handle zero vector case, treat as max distance from anything
    double min_cosine_distance = 2.0; // Cosine distance ranges from 0 to 2
    for (int i = 0; i < NOVELTY_ARCHIVE_SIZE; ++i) {
        const double* archive_vector = &sim->novelty_archive[i * dim];
        double archive_magnitude = vector_magnitude(archive_vector, dim);
        if (archive_magnitude < 1e-9) continue; // Skip zero vectors in archive
        double dot_product = vector_dot_product(query_vector, archive_vector, dim);
        double cosine_similarity = dot_product / (query_magnitude * archive_magnitude);
        // Clamp cosine_similarity to [-1, 1] due to potential floating point inaccuracies
        cosine_similarity = fmax(-1.0, fmin(1.0, cosine_similarity));
        double current_cosine_distance = 1.0 - cosine_similarity;
        if (current_cosine_distance < min_cosine_distance) {
            min_cosine_distance = current_cosine_distance;
        }
    }
    return min_cosine_distance;
}

static void novelty_archive_add(Simulation* sim, double *vector, int dim) {
    memcpy(&sim->novelty_archive[sim->novelty_archive_ptr * dim], vector, dim * sizeof(double));
    sim->novelty_archive_ptr = (sim->novelty_archive_ptr + 1) % NOVELTY_ARCHIVE_SIZE; // Circular buffer
}

static double worker_calculate_taste(Simulation* sim, int cx, int cy, double *worker_latent_context) {
    // 1. Novelty Search Component
    double novelty_score = novelty_archive_get_nn_distance(sim, worker_latent_context, WORKER_LATENT_DIM);
    // 2. Predictive Coding Component (Goldilocks Zone)
    double current_pixel_value = sim->global_canvas[canvas_idx(cx, cy)];
    double previous_pixel_value = sim->previous_global_canvas[canvas_idx(cx, cy)];
    double prediction_error = fabs(current_pixel_value - previous_pixel_value);
    // Accumulate prediction error for adaptive influence logic
#pragma omp atomic
    sim->sum_prediction_error += prediction_error;
#pragma omp atomic
    sim->taste_eval_count++;
    // Gaussian-like function for Goldilocks zone: rewards errors near optimal, penalizes far off
    double predictive_score = exp(-pow(prediction_error - PREDICTIVE_ERROR_OPTIMAL, 2) / (2 * pow(PREDICTIVE_ERROR_TOLERANCE, 2)));
    // Combine components
    double total_taste = (novelty_score * sim->params.taste_novelty_weight) + (predictive_score * sim->params.taste_predictive_weight);
    // Add current context to novelty archive after calculating taste
    novelty_archive_add(sim, worker_latent_context, WORKER_LATENT_DIM);
    return total_taste;
}

static void draw_on_canvas(Simulation* sim, double *canvas, double *target_canvas, double target_influence, int x0, int y0, int x1, int y1, double intensity) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    while (true) {
        int idx = canvas_idx(x0, y0);
        double p = canvas[idx];
        double modified = p + intensity;
        modified = fmax(0.0, fmin(1.0, modified));
        if (target_canvas && target_influence > 0.0) {
            double tp = target_canvas[idx];
            canvas[idx] = target_influence * tp + (1.0 - target_influence) * modified;
        } else {
            canvas[idx] = modified;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

static void worker_step(Simulation* sim, Machine *mach) {
    if (!mach->current || mach->energy <= 0) return;
    mach->energy -= ENERGY_COST_PER_ACTION;
    // Determine which name/ritual to interpret
    const char* name_to_interpret = NULL;
    int current_ritual_idx = sim->ritual_registry.active_idx;
    if (current_ritual_idx != -1) {
        name_to_interpret = sim->ritual_registry.registry[current_ritual_idx];
    } else if (sim->name_registry.count > 0) {
        name_to_interpret = sim->name_registry.registry[sim_xorshift(sim) % sim->name_registry.count]; // stochastic step
    }
    int cx = mach->current->target_x;
    int cy = mach->current->target_y;
    double latent_in[WORKER_LATENT_DIM];
    encode_context(sim->global_canvas, cx, cy, latent_in, WORKER_LATENT_DIM);
    if (current_ritual_idx != -1) {
        // Ritual influence on worker's latent input
        for (int d = 0; d < WORKER_LATENT_DIM; ++d) {
            latent_in[d] += sim->ritual_registry.modulators[current_ritual_idx].meaning_vector[d];
        }
        // Update ritual dissonance and usage context
        double dissonance = latent_distance(latent_in, sim->ritual_registry.modulators[current_ritual_idx].meaning_vector, WORKER_LATENT_DIM);
#pragma omp atomic
        sim->ritual_registry.modulators[current_ritual_idx].accumulated_dissonance += dissonance;
#pragma omp atomic
        sim->ritual_registry.modulators[current_ritual_idx].usage_count++;
        for(int d=0; d<WORKER_LATENT_DIM; d++) {
            sim->ritual_registry.modulators[current_ritual_idx].current_usage_context_avg[d] = (RITUAL_CONTEXT_AVG_ALPHA * sim->ritual_registry.modulators[current_ritual_idx].current_usage_context_avg[d]) + ((1.0 - RITUAL_CONTEXT_AVG_ALPHA) * latent_in[d]);
        }
    }
    // Apply name/ritual modifications if applicable
    double bias_mod = 0.0;
    double angle_mod = 0.0;
    double weight_mod = 0.0;
    double radius_mod = 0.0;
    NameDictionary *dict = &sim->worker_name_dicts[mach->id];
    if (name_to_interpret) {
        for (int i = 0; i < dict->action_count; ++i) {
            if (strcmp(dict->actions[i].name, name_to_interpret) == 0) {
                bias_mod = dict->actions[i].bias_mod;
                angle_mod = dict->actions[i].angle_mod;
                weight_mod = dict->actions[i].weight_mod;
                radius_mod = dict->actions[i].radius_mod;
                break;
            }
        }
    }
    mach->current->bias += bias_mod; // Apply bias modification
    // --- Calculate Guidance Reward BEFORE worker action ---
    double latent_dist_before = 0.0;
    double ritual_dist_before = 0.0;
    double local_mse_before = 0.0; // Reintroduced for image_reward
    const int REWARD_RADIUS = 10;
    for (int y = -REWARD_RADIUS; y <= REWARD_RADIUS; y++) {
        for (int x = -REWARD_RADIUS; x <= REWARD_RADIUS; x++) {
            int current_pixel_idx = canvas_idx(cx + x, cy + y);
            double rl[WORKER_LATENT_DIM];
            encode_context(sim->global_canvas, cx + x, cy + y, rl, WORKER_LATENT_DIM);
            double *gv = &sim->guidance_latent_canvas[canvas_idx(cx + x, cy + y) * MANAGER_LATENT_DIM];
            latent_dist_before += latent_distance(gv, rl, WORKER_LATENT_DIM);
            if (current_ritual_idx != -1) {
                ritual_dist_before += latent_distance(rl, sim->ritual_registry.modulators[current_ritual_idx].goal_vector, WORKER_LATENT_DIM);
            }
            double pixel_diff = sim->global_canvas[current_pixel_idx] - sim->target_canvas[current_pixel_idx];
            local_mse_before += pixel_diff * pixel_diff; // Calculate MSE to target
        }
    }
    double prediction = node_evaluate_in_latent(mach->current, latent_in);
    double confidence_base = sim->params.confidence_base * mach->local_confidence_modifier;
    double confidence_gain = sim->params.confidence_gain * mach->local_confidence_modifier;
    double c = confidence_base + confidence_gain * fabs(prediction);
    c = fmax(0.0, fmin(1.0, c));
    // Select kernel element based on prediction
    int selected = (int)(((prediction + 1.0) / 2.0) * (mach->kernel.count - 1));
    KernelElement elem = mach->kernel.elements[selected];
    elem.angle += angle_mod;
    elem.weight += weight_mod;
    elem.radius += radius_mod;
    double r = elem.radius * mach->worker_imprint_radius;
    int dx = (int)round(cos(elem.angle) * r);
    int dy = (int)round(sin(elem.angle) * r);
    int new_cx = cx + dx;
    int new_cy = cy + dy;
    double intensity = prediction * elem.weight * c;
    // Apply the action: draw line
    draw_on_canvas(sim, sim->global_canvas, sim->target_canvas, sim->target_influence_weight, cx, cy, new_cx, new_cy, intensity);
    // --- Calculate Guidance Reward AFTER worker action ---
    double latent_dist_after = 0.0;
    double ritual_dist_after = 0.0;
    double local_mse_after = 0.0;
    for (int y = -REWARD_RADIUS; y <= REWARD_RADIUS; y++) {
        for (int x = -REWARD_RADIUS; x <= REWARD_RADIUS; x++) {
            int current_pixel_idx = canvas_idx(cx + x, cy + y);
            double rl[WORKER_LATENT_DIM];
            encode_context(sim->global_canvas, cx + x, cy + y, rl, WORKER_LATENT_DIM);
            double *gv = &sim->guidance_latent_canvas[canvas_idx(cx + x, cy + y) * MANAGER_LATENT_DIM];
            latent_dist_after += latent_distance(gv, rl, WORKER_LATENT_DIM);
            if (current_ritual_idx != -1) {
                ritual_dist_after += latent_distance(rl, sim->ritual_registry.modulators[current_ritual_idx].goal_vector, WORKER_LATENT_DIM);
            }
            double pixel_diff = sim->global_canvas[current_pixel_idx] - sim->target_canvas[current_pixel_idx];
            local_mse_after += pixel_diff * pixel_diff;
        }
    }
    // --- Calculate Taste Score for Worker ---
    double taste_score = worker_calculate_taste(sim, cx, cy, latent_in);
    // Calculate reward components
    double guidance_reward = latent_dist_before - latent_dist_after;
    double image_reward = local_mse_before - local_mse_after;
    double ritual_goal_reward = 0.0;
    if (current_ritual_idx != -1) {
        ritual_goal_reward = ritual_dist_before - ritual_dist_after;
    }
    // Total reward for the worker - Bayesian blend of image_reward and taste_score
    double total_reward = (image_reward * sim->target_influence_weight * 0.2) + // Reduce image_reward contribution
                          (taste_score * (1.0 - sim->target_influence_weight) * 2.0) + // Boost taste_score
                          (sim->params.guidance_reward_weight * guidance_reward) +
                          (sim->params.ritual_goal_reward_weight * ritual_goal_reward);
    if (total_reward > 0) {
        mach->energy += total_reward * WORKER_ENERGY_GAIN;
        if (current_ritual_idx != -1) mach->energy += RITUAL_ENERGY_BONUS;
        // Pay a portion of reward to the guiding manager
        int manager_idx = (canvas_idx(cx, cy) * MANAGER_COUNT) / (CANVAS_WIDTH * CANVAS_HEIGHT);
#pragma omp atomic
        sim->managers[manager_idx].energy += total_reward * WORKER_MANAGER_PAYMENT_RATIO;
        // Update name stats if name was interpreted
        if (name_to_interpret) {
            for (int j = 0; j < sim->name_registry.count; j++) {
                if (strcmp(sim->name_registry.registry[j], name_to_interpret) == 0) {
                    sim->name_registry.usage_stats[j].success_count++;
                    sim->name_registry.usage_stats[j].total_reward += total_reward;
                    sim->name_registry.usage_stats[j].uses++;
                    break;
                }
            }
        }
    }
    node_propagate_error_up(mach->current, latent_dist_after);
    ProgramTreeNode *node = mach->current;
    while(node != NULL) {
        if (fabs(node->prediction_error) > HEBBIAN_THRESHOLD) {
            double e = node->prediction_error;
            node->error_variance = (ERROR_VARIANCE_ALPHA * node->error_variance) + ((1.0 - ERROR_VARIANCE_ALPHA) * e * e);
            double learning_rate = sim->params.worker_base_lr * mach->local_lr_modifier;
            if(mach->plasticity_timer > 0) {
                learning_rate *= sim->params.ritual_plasticity_modifier;
            }
            for (int d = 0; d < WORKER_LATENT_DIM; d++) {
                node->w_latent[d] += learning_rate * (e * latent_in[d] / (node->error_variance + latent_in[d]*latent_in[d]));
            }
            node->bias += learning_rate * (e / (node->error_variance + 1.0));
        }
        node->prediction_error = 0.0;
        node = node->parent;
    }
    if (mach->plasticity_timer > 0) mach->plasticity_timer--;
    if (mach->energy < REWIRE_ENERGY_THRESHOLD) machine_rewire(sim, mach);
    mach->current = (prediction < 0) ? mach->current->next_left : mach->current->next_right;
    if (mach->current) {
        mach->current->target_x = new_cx;
        mach->current->target_y = new_cy;
        mach->current->target_x = (mach->current->target_x % CANVAS_WIDTH + CANVAS_WIDTH) % CANVAS_WIDTH;
        mach->current->target_y = (mach->current->target_y % CANVAS_HEIGHT + CANVAS_HEIGHT) % CANVAS_HEIGHT;
    } else {
        mach->current = mach->root;
    }
}

// --- Simulation Core Logic (`simulation_`) ---
// Encapsulates all parameter adaptation logic
static void simulation_adapt_parameters(Simulation* sim) {
    if (sim->step % ADAPTATION_CHECK_INTERVAL != 0) return;
    double current_mse = canvas_evaluate_mse_loss(sim);
    double mse_improvement = 0.0;
    if (sim->mse_at_window_start < 0) {
        sim->mse_at_window_start = current_mse;
    } else {
        mse_improvement = sim->mse_at_window_start - current_mse;
        sim->mse_at_window_start = current_mse; // Update for next interval
    }
    double avg_prediction_error = (sim->taste_eval_count > 0) ? sim->sum_prediction_error / sim->taste_eval_count : 0.0;
    // --- Adaptive Target Influence Logic ---
    // Determine if we are stagnating based on absolute or relative terms
    int is_stagnating = 0;
    // Stagnation if improvement is less than a very small absolute value OR
    // if improvement is less than a small percentage of the current MSE (for higher MSEs)
    if (mse_improvement < STAGNATION_THRESHOLD || (mse_improvement < (current_mse * RELATIVE_STAGNATION_FACTOR) && current_mse > GOOD_ENOUGH_MSE)) {
        is_stagnating = 1;
    }
    if (current_mse > GOOD_ENOUGH_MSE) {
        sim->target_influence_weight -= is_stagnating ? sim->params.target_influence_adapt_rate * 2.0 : sim->params.target_influence_adapt_rate * 0.5;
        char log_msg[256];
        sprintf(log_msg, "MSE: %.6f, Impr: %.6f. Reducing target influence to %.2f (stagnating: %d).", current_mse, mse_improvement, sim->target_influence_weight, is_stagnating);
        simulation_log_event(sim, log_msg);
    } else {
        if (avg_prediction_error > sim->params.taste_influence_threshold_chaos * 2.0) {
            sim->target_influence_weight += sim->params.target_influence_adapt_rate * 0.5;
            char log_msg[256];
            sprintf(log_msg, "Target met, very chaotic (avg pred error %.4f). Slightly increasing target influence to %.2f.", avg_prediction_error, sim->target_influence_weight);
            simulation_log_event(sim, log_msg);
        } else {
            sim->target_influence_weight -= sim->params.target_influence_adapt_rate * 1.5;
            char log_msg[256];
            sprintf(log_msg, "Target met, stable (avg pred error %.4f). Reducing target influence to %.2f.", avg_prediction_error, sim->target_influence_weight);
            simulation_log_event(sim, log_msg);
        }
    }
    // Add random perturbation // stochastic step
    sim->target_influence_weight += (sim_rand_double(sim) - 0.5) * sim->params.random_influence_perturbation;
    // Clamp to [MIN_TARGET_INFLUENCE, 0.8] to cap target influence
    sim->target_influence_weight = fmax(MIN_TARGET_INFLUENCE, fmin(0.8, sim->target_influence_weight));
    // Reset taste tracking for next interval
    sim->sum_prediction_error = 0.0;
    sim->taste_eval_count = 0;
    // --- End Adaptive Target Influence Logic ---
    if (is_stagnating) {
        sim->params.critic_target_weight -= sim->params.target_influence_adapt_rate * 0.5;
        sim->params.critic_target_weight = fmax(0.1, sim->params.critic_target_weight);
        char log_msg[256];
        sprintf(log_msg, "Stagnation detected. Reducing critic target weight to %.2f.", sim->params.critic_target_weight);
        simulation_log_event(sim, log_msg);
    }
    double total_critic_score = 0;
    for (int i = 0; i < CANVAS_WIDTH * CANVAS_HEIGHT; i++) {
        total_critic_score += sim->criticism_canvas[i];
    }
    double current_critic_score = total_critic_score / (CANVAS_WIDTH * CANVAS_HEIGHT);
    if (sim->critic_score_at_window_start < 0) {
        sim->critic_score_at_window_start = current_critic_score;
    } else {
        double critic_improvement = current_critic_score - sim->critic_score_at_window_start;
        if (critic_improvement < AESTHETIC_STAGNATION_THRESHOLD) {
            simulation_log_event(sim, "Aesthetic stagnation! Mutating Critic parameters.");
            sim->params.guidance_reward_weight *= (1.0 + (sim_rand_double(sim) - 0.5) * PARAMETER_MUTATION_STRENGTH); // stochastic step
            sim->params.ritual_goal_reward_weight *= (1.0 + (sim_rand_double(sim) - 0.5) * PARAMETER_MUTATION_STRENGTH); // stochastic step
            sim->params.critic_base_lr *= (1.0 + (sim_rand_double(sim) - 0.5) * PARAMETER_MUTATION_STRENGTH); // stochastic step
            sim->params.critic_novelty_weight *= (1.0 + (sim_rand_double(sim) - 0.5) * PARAMETER_MUTATION_STRENGTH); // stochastic step
            sim->params.critic_repetition_penalty_weight *= (1.0 + (sim_rand_double(sim) - 0.5) * PARAMETER_MUTATION_STRENGTH); // stochastic step
            if (sim->params.guidance_reward_weight < 10) sim->params.guidance_reward_weight = 10;
            if (sim->params.ritual_goal_reward_weight < 5) sim->params.ritual_goal_reward_weight = 5;
            if (sim->params.critic_base_lr < 0.0001) sim->params.critic_base_lr = 0.0001;
            if (sim->params.critic_base_lr > 0.01) sim->params.critic_base_lr = 0.01;
            if (sim->params.critic_novelty_weight < 0.1) sim->params.critic_novelty_weight = 0.1;
            if (sim->params.critic_novelty_weight > 1.0) sim->params.critic_novelty_weight = 1.0;
            if (sim->params.critic_repetition_penalty_weight < 0.1) sim->params.critic_repetition_penalty_weight = 0.1;
            if (sim->params.critic_repetition_penalty_weight > 1.0) sim->params.critic_repetition_penalty_weight = 1.0;
        }
        sim->critic_score_at_window_start = current_critic_score;
    }
}

// Main simulation step for recreation phase
static void simulation_step(Simulation* sim) {
    critic_phase(sim);
    ritual_invocation_phase(sim);
    naming_phase(sim);
    interpretation_phase(sim);
    // Ritualization phase needs to iterate through names, so it's placed here
    for (int i = 0; i < sim->name_registry.count; i++) {
        ritual_promote_name(sim, i);
    }
    ritual_maintenance_phase(sim);
    simulation_adapt_parameters(sim);
#pragma omp parallel for
    for (int m = 0; m < MANAGER_COUNT; m++) {
        manager_step(sim, &sim->managers[m]);
    }
#pragma omp parallel for
    for (int w = 0; w < WORKER_COUNT; w++) {
        worker_step(sim, &sim->workers[w]);
    }
    memcpy(sim->previous_global_canvas, sim->global_canvas, CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
}

// Main simulation step for pattern writing phase
static void simulation_pattern_writing_step(Simulation* sim) {
    critic_phase(sim); // Critics still run, judging the Pattern_Canvas
#pragma omp parallel for
    for (int w = 0; w < WORKER_COUNT; w++) {
        Machine *mach = &sim->workers[w];
        if (!mach->current) continue;
        int cx = mach->current->target_x;
        int cy = mach->current->target_y;
        // In this phase, workers just apply their learned patterns without external rewards
        // and no energy consumption/learning/rewiring. They are in 'performance' mode.
        double latent_in[WORKER_LATENT_DIM];
        memset(latent_in, 0, sizeof(double) * WORKER_LATENT_DIM); // No external context influence
        double prediction = node_evaluate_in_latent(mach->current, latent_in);
        int selected = (int)(((prediction + 1.0) / 2.0) * (mach->kernel.count - 1));
        KernelElement elem = mach->kernel.elements[selected];
        double r = elem.radius * mach->worker_imprint_radius;
        int dx = (int)round(cos(elem.angle) * r);
        int dy = (int)round(sin(elem.angle) * r);
        int new_cx = cx + dx;
        int new_cy = cy + dy;
        double intensity = prediction * elem.weight * 0.5; // Fixed confidence
        // Apply the action: draw line (no blend in pattern mode)
        draw_on_canvas(sim, sim->pattern_canvas, NULL, 0.0, cx, cy, new_cx, new_cy, intensity);
        mach->current = (prediction < 0) ? mach->current->next_left : mach->current->next_right;
        if (mach->current) {
            mach->current->target_x = new_cx;
            mach->current->target_y = new_cy;
            mach->current->target_x = (mach->current->target_x % CANVAS_WIDTH + CANVAS_WIDTH) % CANVAS_WIDTH;
            mach->current->target_y = (mach->current->target_y % CANVAS_HEIGHT + CANVAS_HEIGHT) % CANVAS_HEIGHT;
        } else {
            mach->current = mach->root;
        }
    }
    memcpy(sim->previous_pattern_canvas, sim->pattern_canvas, CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
}

// --- Visualization, Cleanup, & Main ---
// Save canvas to PNG
static void save_canvas_snapshot(const char* filename, const double* canvas_to_save) {
    printf("Saving image to %s...\n", filename);
    unsigned char* data = (unsigned char*)malloc(CANVAS_WIDTH * CANVAS_HEIGHT);
    if(!data) {
        perror("Failed to allocate memory for image data");
        return;
    }
    for (int i = 0; i < CANVAS_WIDTH * CANVAS_HEIGHT; i++) {
        double v = fmax(0.0, fmin(1.0, canvas_to_save[i]));
        data[i] = (unsigned char)(v * 255.0);
    }
    stbi_write_png(filename, CANVAS_WIDTH, CANVAS_HEIGHT, 1, data, CANVAS_WIDTH);
    free(data);
    printf("Saved to %s\n", filename);
}

// Log event to file
static void simulation_log_event(Simulation* sim, const char* message) {
    if (sim->event_log_file) {
        fprintf(sim->event_log_file, "[Step %d] %s\n", sim->step, message);
    }
}

// Free machine nodes (helper for cleanup)
static void free_machine_nodes(Machine *mach) {
    if (mach->all_nodes) {
        for(int j=0; j<mach->node_count; ++j) {
            if(mach->all_nodes[j]) {
                free(mach->all_nodes[j]->w_latent);
                free(mach->all_nodes[j]);
            }
        }
        free(mach->all_nodes);
    }
}

// Cleanup function to free all allocated memory and close files
static void simulation_cleanup(Simulation* sim) {
    printf("Cleaning up...\n");
    save_canvas_snapshot("final_output.png", sim->global_canvas);
    if (sim->pattern_writing_mode || sim->pattern_step > 0) {
        save_canvas_snapshot("final_patterns.png", sim->pattern_canvas);
    }
    if (sim->log_file) fclose(sim->log_file);
    if (sim->event_log_file) fclose(sim->event_log_file);
    free(sim->global_canvas);
    free(sim->target_canvas);
    free(sim->guidance_latent_canvas);
    free(sim->criticism_canvas);
    free(sim->previous_global_canvas);
    free(sim->pattern_canvas);
    free(sim->previous_pattern_canvas);
    free(sim->novelty_archive);
    for(int i=0; i<sim->name_registry.count; ++i) {
        if(sim->name_registry.registry[i]) free(sim->name_registry.registry[i]);
        if(sim->name_registry.usage_stats[i].name) free(sim->name_registry.usage_stats[i].name);
    }
    for(int i=0; i<sim->ritual_registry.count; ++i) if(sim->ritual_registry.registry[i]) free(sim->ritual_registry.registry[i]);
    for(int i=0; i<WORKER_COUNT; i++) {
        for(int j=0; j<sim->worker_name_dicts[i].action_count; ++j) {
            if(sim->worker_name_dicts[i].actions[j].name) free(sim->worker_name_dicts[i].actions[j].name);
        }
    }
    for(int i=0; i<CRITIC_COUNT; i++) free_machine_nodes(&sim->critics[i]);
    for(int i=0; i<MANAGER_COUNT; i++) free_machine_nodes(&sim->managers[i]);
    for(int i=0; i<WORKER_COUNT; i++) free_machine_nodes(&sim->workers[i]);
    printf("Cleanup complete.\n");
}

// GLUT display callback
static void display_canvas(void) {
    // We need a static pointer to Simulation to pass to GLUT callbacks
    // This is a common pattern when refactoring globals into a struct for GLUT
    static Simulation* current_sim_ptr = NULL;
    if (g_sim_instance != NULL) { // Use the global instance
        current_sim_ptr = g_sim_instance;
    } else {
        return; // Should not happen if initialized correctly
    }
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glBegin(GL_POINTS);
    const double* current_display_canvas = current_sim_ptr->pattern_writing_mode ? current_sim_ptr->pattern_canvas : current_sim_ptr->global_canvas;
    for (int y = 0; y < CANVAS_HEIGHT; y++) {
        for (int x = 0; x < CANVAS_WIDTH; x++) {
            double val = current_display_canvas[canvas_idx(x, y)];
            glColor3f(val, val, val);
            glVertex2i(x, y);
        }
    }
    glEnd();
    glutSwapBuffers();
    char title[256];
    if (current_sim_ptr->pattern_writing_mode) {
        sprintf(title, "Pattern Writing Mode | Step: %d | Patterns: %d", current_sim_ptr->pattern_step, WORKER_COUNT);
    } else {
        double loss = canvas_evaluate_mse_loss(current_sim_ptr);
        sprintf(title, "Hybrid Sim (Adaptive) | Step: %d | Target Inf: %.2f | MSE: %.6f | Names: %d | Rituals: %d", current_sim_ptr->step, current_sim_ptr->target_influence_weight, loss, current_sim_ptr->name_registry.count, current_sim_ptr->ritual_registry.count);
    }
    glutSetWindowTitle(title);
}

// GLUT idle callback
static void update_simulation(void) {
    // We need a static pointer to Simulation to pass to GLUT callbacks
    static Simulation* current_sim_ptr = NULL;
    if (g_sim_instance != NULL) { // Use the global instance
        current_sim_ptr = g_sim_instance;
    } else {
        return; // Should not happen if initialized correctly
    }
    if (!current_sim_ptr->pattern_writing_mode) {
        if (current_sim_ptr->step < MAX_SIM_STEPS) {
            simulation_step(current_sim_ptr);
            if (current_sim_ptr->step % 100 == 0) {
                glutPostRedisplay();
            }
            if (current_sim_ptr->step % 10 == 0 && current_sim_ptr->log_file) {
                fprintf(current_sim_ptr->log_file, "%d,%.6f\n", current_sim_ptr->step, canvas_evaluate_mse_loss(current_sim_ptr));
            }
            current_sim_ptr->step++;
        } else {
            // Transition to pattern writing mode
            current_sim_ptr->pattern_writing_mode = 1;
            current_sim_ptr->pattern_step = 0;
            printf("--- Transitioning to Pattern Writing Mode ---\n");
            simulation_log_event(current_sim_ptr, "Transitioning to Pattern Writing Mode.");
            // Initialize Previous_Pattern_Canvas to the initial white state
            for (int i = 0; i < CANVAS_WIDTH * CANVAS_HEIGHT; ++i) {
                current_sim_ptr->previous_pattern_canvas[i] = 1.0;
            }
            // Optionally reset worker positions to start drawing from a consistent place
#pragma omp parallel for // stochastic step
            for (int i = 0; i < WORKER_COUNT; ++i) {
                current_sim_ptr->workers[i].current = current_sim_ptr->workers[i].root;
                current_sim_ptr->workers[i].current->target_x = sim_xorshift(current_sim_ptr) % CANVAS_WIDTH;
                current_sim_ptr->workers[i].current->target_y = sim_xorshift(current_sim_ptr) % CANVAS_HEIGHT;
            }
            glutPostRedisplay(); // Force immediate display update
        }
    } else {
        // In pattern writing mode
        if (current_sim_ptr->pattern_step < MAX_PATTERN_WRITING_STEPS) {
            simulation_pattern_writing_step(current_sim_ptr);
            if (current_sim_ptr->pattern_step % 100 == 0) {
                glutPostRedisplay();
            }
            // Save snapshot every 1000 steps in pattern writing mode
            if (current_sim_ptr->pattern_step > 0 && current_sim_ptr->pattern_step % 1000 == 0) {
                char filename[64];
                sprintf(filename, "pattern_snapshot_%05d.png", current_sim_ptr->pattern_step);
                save_canvas_snapshot(filename, current_sim_ptr->pattern_canvas);
                char log_msg[128];
                sprintf(log_msg, "Saved pattern snapshot %s", filename);
                simulation_log_event(current_sim_ptr, log_msg);
            }
            current_sim_ptr->pattern_step++;
        } else {
            // Pattern writing phase complete
            printf("--- Pattern Writing Mode Complete ---\n");
            simulation_log_event(current_sim_ptr, "Pattern Writing Mode Complete.");
            // Optionally exit GLUT loop when simulation ends
            // glutLeaveMainLoop();
        }
    }
}

// Global pointer to the simulation instance for GLUT callbacks
// static Simulation* g_sim_instance = NULL; // Already declared at top

// GLUT exit handler wrapper
void glut_exit_handler(void) {
    if (g_sim_instance) {
        simulation_cleanup(g_sim_instance);
        free(g_sim_instance); // Free the simulation instance itself
        g_sim_instance = NULL;
    }
}

// Load target image into sim->target_canvas
static int load_target_image(Simulation* sim) {
    int w, h, c;
    unsigned char *img = stbi_load(TARGET_IMAGE_FILENAME, &w, &h, &c, 0);
    if (img == NULL) {
        printf("Error loading image '%s': %s\n", TARGET_IMAGE_FILENAME, stbi_failure_reason());
        simulation_log_event(sim, "Error loading target image.");
        return 0;
    }
    if (w != CANVAS_WIDTH || h != CANVAS_HEIGHT) {
        printf("Error: Target image dimensions (%dx%d) do not match canvas dimensions (%dx%d).\n", w, h, CANVAS_WIDTH, CANVAS_HEIGHT);
        simulation_log_event(sim, "Target image dimensions mismatch.");
        stbi_image_free(img);
        return 0;
    }
    for (int y = 0; y < CANVAS_HEIGHT; y++) {
        for (int x = 0; x < CANVAS_WIDTH; x++) {
            unsigned char* p = img + (x + y * w) * c;
            double r = (double)p[0]/255.0;
            double g = (c > 1)?(double)p[1]/255.0:r;
            double b = (c > 2)?(double)p[2]/255.0:r;
            sim->target_canvas[canvas_idx(x, y)] = 0.299*r + 0.587*g + 0.114*b;
        }
    }
    stbi_image_free(img);
    printf("Loaded target image '%s'.\n", TARGET_IMAGE_FILENAME);
    simulation_log_event(sim, "Target image loaded successfully.");
    return 1;
}

// Initialize default system parameters
static void init_default_params(Simulation* sim) {
    sim->params.confidence_base = 0.01;
    sim->params.confidence_gain = 0.25;
    sim->params.reward_influence = 0.25;
    sim->params.worker_base_lr = 0.01;
    sim->params.worker_kernel_lr = 0.01;
    sim->params.guidance_reward_weight = 100.0;
    sim->params.ritual_goal_reward_weight = 50.0;
    sim->params.ritual_plasticity_modifier = 2.5;
    sim->params.critic_base_lr = 0.005;
    sim->params.critic_target_weight = 0.3; // Reduce critic focus on target
    sim->params.critic_novelty_weight = 0.5;
    sim->params.critic_repetition_penalty_weight = 0.5;
    sim->params.critic_pattern_novelty_weight = 1.0;
    sim->params.critic_pattern_repetition_penalty_weight = 1.0;
    sim->params.taste_novelty_weight = 100.0; // Increased from 50.0
    sim->params.taste_predictive_weight = 40.0; // Increased from 20.0
    sim->params.target_influence_adapt_rate = 0.05;
    sim->params.taste_influence_threshold_chaos = 0.07;
    sim->params.random_influence_perturbation = 0.01;
    // Ritual imprint radius increase
    sim->params.ritual_imprint_radius_increase = 1;
}

// Initialize the main Simulation struct
static Simulation* simulation_init(void) {
    Simulation* sim = (Simulation*)malloc(sizeof(Simulation));
    if (!sim) {
        perror("Failed to allocate Simulation struct");
        return NULL;
    }
    // Initialize all pointers to NULL for safety
    sim->global_canvas = NULL;
    sim->target_canvas = NULL;
    sim->guidance_latent_canvas = NULL;
    sim->criticism_canvas = NULL;
    sim->previous_global_canvas = NULL;
    sim->pattern_canvas = NULL;
    sim->previous_pattern_canvas = NULL;
    sim->novelty_archive = NULL;
    sim->log_file = NULL;
    sim->event_log_file = NULL;
    // Initialize parameters and state
    init_default_params(sim);
    sim->step = 0;
    sim->pattern_step = 0;
    sim->pattern_writing_mode = 0;
    sim->target_influence_weight = 0.5; // Start with less target influence
    sim->mse_at_window_start = -1.0;
    sim->critic_score_at_window_start = -1.0;
    sim->sum_prediction_error = 0.0;
    sim->taste_eval_count = 0;
    // Initialize RNG state
    sim->xorshift_state = (uint64_t)time(NULL) ^ (uint64_t)getpid(); // stochastic step
    // Allocate canvases
    sim->global_canvas = (double*)malloc(CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
    sim->target_canvas = (double*)malloc(CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
    sim->guidance_latent_canvas = (double*)malloc(CANVAS_WIDTH * CANVAS_HEIGHT * MANAGER_LATENT_DIM * sizeof(double));
    sim->criticism_canvas = (double*)malloc(CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
    sim->previous_global_canvas = (double*)malloc(CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
    sim->pattern_canvas = (double*)malloc(CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
    sim->previous_pattern_canvas = (double*)malloc(CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
    sim->novelty_archive = (double*)malloc(NOVELTY_ARCHIVE_SIZE * WORKER_LATENT_DIM * sizeof(double));
    if (!sim->global_canvas || !sim->target_canvas || !sim->guidance_latent_canvas || !sim->criticism_canvas || !sim->previous_global_canvas || !sim->pattern_canvas || !sim->previous_pattern_canvas || !sim->novelty_archive) {
        perror("Failed to allocate canvas or archive memory");
        simulation_cleanup(sim); // Clean up partially allocated memory
        free(sim); // Free the sim struct itself
        return NULL;
    }
    // Load target image
    if (!load_target_image(sim)) {
        simulation_cleanup(sim);
        free(sim);
        return NULL;
    }
    // Initialize canvases
    canvas_randomize(sim, sim->global_canvas);
    memcpy(sim->previous_global_canvas, sim->global_canvas, CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
    memset(sim->criticism_canvas, 0, CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(double));
    for (int i = 0; i < CANVAS_WIDTH * CANVAS_HEIGHT; ++i) {
        sim->pattern_canvas[i] = 1.0;
        sim->previous_pattern_canvas[i] = 1.0;
    }
    novelty_archive_init(sim);
    // Initialize critics, managers, workers
    for (int i = 0; i < CRITIC_COUNT; i++) critic_init(sim, &sim->critics[i], i);
    for (int i = 0; i < MANAGER_COUNT; i++) manager_init(sim, &sim->managers[i], i);
    for (int i = 0; i < WORKER_COUNT; i++) worker_init(sim, &sim->workers[i], i);
    // Initialize name and ritual registries
    naming_init(sim);
    ritual_init(sim);
    // Open log files
    sim->log_file = fopen("hybrid_sim_loss_log.csv", "w");
    if (!sim->log_file) {
        perror("Could not open loss log file");
        return NULL;
    }
    fprintf(sim->log_file, "Step,Loss\n");
    sim->event_log_file = fopen("hybrid_sim_events.log", "w");
    if (!sim->event_log_file) {
        perror("Could not open event log file");
        return NULL;
    }
    simulation_log_event(sim, "Simulation initialized.");
    return sim;
}

int main(int argc, char **argv) {
    // Initialize the main simulation struct
    g_sim_instance = simulation_init();
    if (g_sim_instance == NULL) {
        return 1; // Exit if initialization failed
    }
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(CANVAS_WIDTH, CANVAS_HEIGHT);
    glutCreateWindow("Hybrid Adaptive Evolutionary Art Generation System");
    glutDisplayFunc(display_canvas); // Register display callback
    glutIdleFunc(update_simulation); // Register idle callback
    atexit(glut_exit_handler); // Register cleanup function to run on exit
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, CANVAS_WIDTH, CANVAS_HEIGHT, 0);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    printf("--- RUNNING HYBRID ADAPTIVE EVOLUTIONARY ART GENERATION SYSTEM ---\n");
    printf("Workers are guided by internal 'taste' (novelty & predictability) and manager guidance.\n");
    printf("Critics adapt to current current phase (recreation vs. pattern writing).\n");
    printf("Target influence on workers is adaptive based on performance.\n");
    printf("Using %d threads.\n", omp_get_max_threads());
    glutMainLoop(); // Start the GLUT event loop
    return 0;
}
