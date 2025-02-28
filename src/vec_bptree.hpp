#ifndef VECBPLUSTREE_HPP
#define VECBPLUSTREE_HPP

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "linear_model/model.hpp"
#include "thread_pool/ThreadPool.hpp"
#ifdef _OPENMP
#pragma message("Compile vec_tree.hpp with OpenMP")
#include <omp.h>
#else
#pragma message("Compilation with -fopenmp is optional but recommended")
// omp_set_num_threads(4);
#endif

#ifndef FANOUT
#define FANOUT 256
#endif

#ifndef MAX_EPSILON
#define MAX_EPSILON 254
#endif

#ifndef MODEL_NODE_MIN_SIZE_FACTOR
#define MODEL_NODE_MIN_SIZE_FACTOR 0.5
#endif

#ifndef MODEL_NODE_MAX_SIZE_FACTOR
#define MODEL_NODE_MAX_SIZE_FACTOR FANOUT
#endif

// #define NODEBUG



template<typename Key, typename Value>
class VecBPlusTree {
public:
    typedef Key key_type;

    typedef Value value_type;

    typedef VecBPlusTree Self;

    typedef std::pair<key_type, value_type> kv_pair;

    //! \name Static Constant Options and Values of the B+ Tree
    //! \{

    //! Base B+ tree parameter: The number of key/data slots in each leaf
    static const unsigned int leaf_slotmax = FANOUT;
    static const unsigned int model_leaf_slotmax = FANOUT * MODEL_NODE_MAX_SIZE_FACTOR;

    //! Base B+ tree parameter: The number of key slots in each inner node,
    //! this can differ from slots in each leaf.
    //! child number is slotmax + 1array_lower_bound
    static const unsigned int inner_slotmax = FANOUT - 1;

    //! Computed B+ tree parameter: The minimum number of key/data slots used
    //! in a leaf. If fewer slots are used, the leaf will be merged or slots
    //! shifted from it's siblings.
    static const unsigned int leaf_slotmin = (leaf_slotmax / 2);
    static const unsigned int model_leaf_slotmin = FANOUT * MODEL_NODE_MIN_SIZE_FACTOR;

    //! Computed B+ tree parameter: The minimum number of key slots used
    //! in an inner node. If fewer slots are used, the inner node will be
    //! merged or slots shifted from it's siblings.
    static const unsigned int inner_slotmin = (inner_slotmax / 2);

    //! The code does linear search in find_lower() and
    //! find_upper() instead of binary_search, unless the node size is larger
    //! than this threshold.
    static const size_t binsearch_threshold = 32;

    struct Statistics {
        size_t num_inner_nodes = 0;
        size_t num_leaf_nodes = 0;
        size_t num_inner_key_slots = 0;
        size_t num_inner_ptr_slots = 0;
        size_t num_leaf_slots = 0;
    };

public:
    struct Node {
        //! Level in the b-tree, if level == 0 -> leaf node
        uint16_t level = 0;

        Node* parent_node = nullptr;

        bool is_locked = false;

        explicit Node(unsigned short level) {
            this->level = level;
            this->parent_node = nullptr;
        }

        //! True if this is a leaf node.
        [[nodiscard]] bool is_leafnode() const { return (level == 0); }

        //! True if this is a model-based leaf node.
        [[nodiscard]] virtual bool is_model_leafnode() const = 0;

        virtual uint32_t slotuse() const = 0;
        virtual uint32_t remain() const = 0;
        virtual const key_type& key(size_t s) const = 0;
        virtual bool is_full() const = 0;
        virtual bool is_few() const = 0;
        virtual bool is_underflow() const = 0;
        virtual const key_type* get_keyslots_ref() const = 0;
        virtual size_t mem_size() const = 0;
        virtual Node* get_child(size_t s) = 0;
    };

    struct InnerNode : Node {
        std::vector<key_type> slotkey;
        std::vector<Node*> child_nodes;

        explicit InnerNode(uint16_t level) : Node(level) {
            slotkey.reserve(inner_slotmin + inner_slotmin / 2);
            child_nodes.reserve(inner_slotmin + inner_slotmin / 2 + 1);
        }

        [[nodiscard]] bool is_model_leafnode() const override { return false; }

        uint32_t slotuse() const override { return slotkey.size(); }

        uint32_t remain() const override { return inner_slotmax - slotkey.size(); }

        const key_type& key(size_t s) const override { return slotkey[s]; }

        bool is_full() const override { return slotkey.size() == inner_slotmax; }

        bool is_few() const override { return slotkey.size() <= inner_slotmin; }

        bool is_underflow() const override { return slotkey.size() < inner_slotmin; }
        // Return the memory size of the node (including heap memory)
        size_t mem_size() const override {
            return sizeof(InnerNode) + slotkey.size() * sizeof(key_type) + child_nodes.size() * sizeof(Node*);
        }

        const key_type* get_keyslots_ref() const override { return slotkey.data(); }

        Node* get_child(size_t s) override { return child_nodes[s]; }

    };

    struct LeafNode : Node {
        //! Double linked list pointers to traverse the leaves
        LeafNode* prev_leaf = nullptr;

        //! Double linked list pointers to traverse the leaves
        LeafNode* next_leaf = nullptr;

        std::vector<key_type> slotdata;

        LeafNode() : Node(0) { slotdata.reserve(leaf_slotmin + leaf_slotmin / 2); }

        virtual ~LeafNode() = default;

        bool insert_data(kv_pair& kv) {
            auto slotkey_ref = this->slotdata.data();
            int insert_idx = array_lower_bound(slotkey_ref, slotuse(), kv.first);
            slotdata.insert(slotdata.begin() + insert_idx, kv.first);

            return true;
        }

        [[nodiscard]] bool is_model_leafnode() const override { return false; }

        void erase_data(key_type key) {
            uint32_t slot = find_lower(this, key);
            if (slot < slotuse() && key == this->key(slot)) {
                slotdata.erase(slotdata.begin() + slot);
            }
        }

        const key_type& key(size_t s) const override { return slotdata[s]; }

        uint32_t slotuse() const override { return slotdata.size(); }

        uint32_t remain() const override { return leaf_slotmax - slotdata.size(); }

        bool is_full() const override { return slotdata.size() == leaf_slotmax; }

        bool is_few() const override { return slotdata.size() <= leaf_slotmin; }

        bool is_underflow() const override { return slotdata.size() < leaf_slotmin; }

        size_t mem_size() const override { return sizeof(LeafNode) + slotdata.size() * sizeof(key_type) + slotdata.size() * sizeof(value_type); }

        const key_type* get_keyslots_ref() const override { return slotdata.data(); }




        Node* get_child(size_t s) override { return nullptr; }

        virtual bool key_exist(const key_type& key) {
            int idx = array_lower_bound(this->slotdata.data(), this->slotuse(), key);
            return this->slotdata[idx] == key;
        }
    };

    struct LearnedLeafNode : LeafNode {
        std::vector<key_type> buffer_slotdata;

        uint32_t buffer_retain_threshold = model_leaf_slotmax;
        uint32_t delete_retain_threshold = model_leaf_slotmax;

        uint32_t delete_cnt = 0;

        bool is_model_ready = false;

        key_type model_original_x = 0xffffffff;
        double model_slope = INTMAX_MAX;
        int64_t model_intercept = INTMAX_MAX;

        LinearModel<key_type, unsigned int>* model = nullptr;

        LearnedLeafNode() : LeafNode() {
            model = new LinearModel<key_type, unsigned int>(MAX_EPSILON);
        }

        explicit LearnedLeafNode(uint32_t buffer_size) : LeafNode() {
            buffer_slotdata.reserve(buffer_size);
            model = new LinearModel<key_type, unsigned int>(MAX_EPSILON);
        }

        ~LearnedLeafNode() override { delete model; }

        void delete_model() {
            delete model;
            model = nullptr;
            is_model_ready = false;
        }

        void get_model_param() {
            auto model = this->model->opt->get_segment();
            auto origin = model.get_first_x();
            auto [slope, intercept] = model.get_floating_point_segment(origin);
            model_slope = slope;
            model_intercept = intercept;
            model_original_x = origin;
            is_model_ready = true;
        }

        void insert_data_to_buffer(kv_pair& kv) {
          buffer_slotdata.emplace_back(kv.first);
        }

        void retrain_new_model() {
            auto new_model = new LinearModel<key_type, unsigned int>(MAX_EPSILON);
            for (unsigned int i = 0; i < this->slotuse(); i++) {
                new_model->add_point(this->key(i), i);
            }
            model = new_model;
            get_model_param();
            is_model_ready = true;
        }

        void retrain() {
            delete model;
            model = nullptr;
            is_model_ready = false;

            std::thread retrain_thread(&LearnedLeafNode::retrain_new_model, this);
            retrain_thread.detach();
        }

        bool key_exist(const key_type& key) override {
            // if (is_model_ready) {
            //     int p_key_index = (key - model_original_x) * model_slope + model_intercept;
            //     p_key_index = p_key_index > 0 ? p_key_index : 0;
            //     p_key_index = p_key_index < this->slotuse() - 1 ? p_key_index : this->slotuse() - 1;
            //
            //     unsigned int search_min = p_key_index < MAX_EPSILON + 1 ? 0 : p_key_index - MAX_EPSILON - 1;
            //     unsigned int search_max = p_key_index + MAX_EPSILON + 1 < this->slotuse() - 1
            //                                       ? p_key_index + MAX_EPSILON + 1
            //                                       : this->slotuse() - 1;
            //     return search_min +
            //           array_lower_bound(this->slotdata.data() + search_min, search_max - search_min + 1, key);
            // } else {
            //     return array_lower_bound(this->slotdata.data(), this->slotuse(), key);
            // }
            int model_idx;
            if (is_model_ready) {
                model_idx = search_model(key);
            } else {
                model_idx = array_lower_bound(this->slotdata.data(), this->slotuse(), key);
            }

            return this->slotdata[model_idx] == key || std::find(buffer_slotdata.begin(), buffer_slotdata.end(), key) != buffer_slotdata.end();
        }

        int search_model(const key_type& key) {
            int p_key_index = (key - model_original_x) * model_slope + model_intercept;
            p_key_index = p_key_index > 0 ? p_key_index : 0;
            p_key_index = p_key_index < this->slotdata.size() - 1 ? p_key_index : this->slotdata.size() - 1;

            unsigned int search_min = p_key_index < MAX_EPSILON + 1 ? 0 : p_key_index - MAX_EPSILON - 1;
            unsigned int search_max = p_key_index + MAX_EPSILON + 1 < this->slotdata.size() - 1
                                              ? p_key_index + MAX_EPSILON + 1
                                              : this->slotdata.size() - 1;
            return search_min +
                array_lower_bound(this->slotdata.data() + search_min, search_max - search_min + 1, key);

        }

        size_t buffer_size() const { return buffer_slotdata.size(); }

        [[nodiscard]] bool is_model_leafnode() const override {
            return is_model_ready;
        }

        const key_type& key(size_t s) const override {
            if (s < this->slotdata.size()) {
                return this->slotdata[s];
            } else {
                return buffer_slotdata[s - this->slotdata.size()];
            }
        }

        uint32_t slotuse() const override { return this->slotdata.size() + buffer_slotdata.size() - delete_cnt; }

        uint32_t remain() const override { return 2 * model_leaf_slotmax - slotuse(); }

        bool is_full() const override { return this->slotdata.size() >= model_leaf_slotmax; }

        bool is_few() const override { return this->slotdata.size() <= model_leaf_slotmin; }

        bool is_underflow() const override {
            return this->slotdata.size() + buffer_slotdata.size() < model_leaf_slotmin;
        }

        size_t mem_size() const override {
            return sizeof(LearnedLeafNode) + this->slotdata.size() * sizeof(key_type) +
                   buffer_slotdata.size() * sizeof(key_type) + this->slotdata.size() * sizeof(value_type) + buffer_slotdata.size() * sizeof(value_type);
        }
    };



public:
    // Tree Object Data Members
    Node* root = nullptr;

    LeafNode* head_leaf = nullptr;
    LeafNode* tail_leaf = nullptr;

    std::vector<key_type> root_buffer;

    Statistics stats;

public:
    explicit VecBPlusTree() {
        root = nullptr;
        head_leaf = nullptr;
        tail_leaf = nullptr;
        root_buffer.reserve(FANOUT * 5);
    }

    ~VecBPlusTree() {
        if (root) {
            std::queue<Node*> q;
            q.push(root);
            while (!q.empty()) {
                auto node = q.front();
                q.pop();
                if (!node->is_leafnode()) {
                    auto inner_node = static_cast<InnerNode*>(node);
                    for (auto child: inner_node->child_nodes) {
                        q.push(child);
                    }
                }
                delete node;
            }
        }
    }

    uint16_t height() { return root->level + 1; }

    void free_node(Node* n) {
        if (n == nullptr) return;
        if (n->is_leafnode()) {
            if (n->is_model_leafnode()) {
                auto* ln = static_cast<LearnedLeafNode*>(n);
                delete ln;
            } else {
                auto* ln = static_cast<LeafNode*>(n);
                delete ln;
            }
        } else {
            auto* in = static_cast<InnerNode*>(n);
            delete in;
        }
    }

    static int find_lower(const Node* n, const key_type& key) {
        auto slotkey_ref = n->get_keyslots_ref();
        return array_lower_bound(slotkey_ref, n->slotuse(), key);

        // if (n->slotuse() >= binsearch_threshold) {
        //     auto it = std::lower_bound(slotkey_ref.begin(), slotkey_ref.end(), key);
        //     if (it != slotkey_ref.end()) {
        //         return it - slotkey_ref.begin();
        //     } else {
        //         return n->slotuse();
        //     }
        // } else {
        //     unsigned int lo = 0;
        //     while (lo < n->slotuse() && n->key(lo) < key)
        //         ++lo;
        //     return lo;
        // }
    }

    LeafNode* find_leaf(const key_type& key) const {
        Node* n = root;
        if (!n) return nullptr;

        while (!n->is_leafnode()) {
            InnerNode* inner = static_cast<InnerNode*>(n);
            int slot = find_lower(inner, key);

            n = inner->child_nodes[slot];
        }

        return static_cast<LeafNode*>(n);
    }

private:
    static int array_lower_bound(const key_type* arr, size_t size, const key_type& key) {
        int lo = 0;
        int hi = size;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (arr[mid] < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

public:
    std::vector<key_type> range_search(const key_type& start_key, const int range_size) {
        std::vector<key_type> results(range_size);
        key_type masked_start_key = start_key << 1;
        Node* n = root;
        if (!n) return results;

        while (!n->is_leafnode()) {
            int slot = find_lower(n, masked_start_key);
            n = n->get_child(slot);
        }
        LeafNode* leaf = static_cast<LeafNode*>(n);
        int i = 0;
        if (leaf->is_model_leafnode()) {
            LearnedLeafNode* learned_leaf = static_cast<LearnedLeafNode*>(leaf);
            i = learned_leaf->search_model(masked_start_key);
        } else {
            i = find_lower(leaf, masked_start_key);
        }
        int cnt = 0;
        while (cnt < range_size && leaf) {
            int to_copy = leaf->slotdata.size() - i > range_size - cnt ? range_size - cnt : leaf->slotdata.size() - i;
            // std::copy(std::make_move_iterator(leaf->slotdata.begin() + i),
            // std::make_move_iterator(leaf->slotdata.begin() + i + end_iter),
            // std::back_inserter(results));

            std::memcpy(results.data() + cnt, leaf->slotdata.data() + i, to_copy * sizeof(key_type));

            cnt += to_copy;
            leaf = leaf->next_leaf;
            i = 0;
        }

        results.resize(cnt);

        return results;
    }

    bool exists(const key_type& search_key) const {
        key_type masked_key = search_key << 1;
        Node* n = root;
        if (!n) return false;

        while (!n->is_leafnode()) {
            int slot = find_lower(n, masked_key);
            n = n->get_child(slot);
        }

        return static_cast<LeafNode *>(n)->key_exist(masked_key);
    }

    bool insert(kv_pair& key_value) {
        bool is_blocked = false;
        kv_pair kv = std::make_pair(key_value.first << 1, key_value.second);
        if (!root) {
            auto new_leaf = new LeafNode();
            root = new_leaf;
            head_leaf = static_cast<LeafNode*>(root);
            tail_leaf = static_cast<LeafNode*>(root);

            new_leaf->insert_data(kv);
            ++stats.num_leaf_nodes;
            ++stats.num_leaf_slots;
            return true;
        }

        Node* curr = root;
        while (!curr->is_leafnode()) {
            InnerNode* inner = static_cast<InnerNode*>(curr);
            is_blocked = inner->is_locked;
            if (is_blocked) {
                root_buffer.emplace_back(kv.first);
                return true;
            }
            int slot = find_lower(inner, kv.first);
            curr = inner->child_nodes[slot];
        }

        LeafNode* leaf = static_cast<LeafNode*>(curr);
        if (!leaf->is_model_leafnode()) {
            if (!leaf->is_full()) {
                ++stats.num_leaf_slots;
                return leaf->insert_data(kv);
            }
            auto insert_pos = find_lower(leaf, kv.first);
            bool insert_left = insert_pos < (leaf->slotuse() >> 1);
            auto [left_leaf, right_leaf] = split_leaf_node(leaf);
            if (insert_left) {
                left_leaf->insert_data(kv);
            } else {
                right_leaf->insert_data(kv);
            }
            ++stats.num_leaf_nodes;
            ++stats.num_leaf_slots;
            insert_new_node(right_leaf, static_cast<InnerNode*>(leaf->parent_node), true,
                            left_leaf->key(left_leaf->slotuse() - 1));
        }

        return true;
    }

    bool insert(key_type& key) {
        bool is_blocked = false;
        kv_pair kv = std::make_pair(key << 1, 0);
        if (!root) {
            auto new_leaf = new LeafNode();
            root = new_leaf;
            head_leaf = static_cast<LeafNode*>(root);
            tail_leaf = static_cast<LeafNode*>(root);

            new_leaf->insert_data(kv);
            return true;
        }

        Node* curr = root;
        while (!curr->is_leafnode()) {
            InnerNode* inner = static_cast<InnerNode*>(curr);
            is_blocked = inner->is_locked;
            if (is_blocked) {
                root_buffer.emplace_back(kv.first);
                return true;
            }
            int slot = find_lower(inner, kv.first);
            curr = inner->child_nodes[slot];
        }

        LeafNode* leaf = static_cast<LeafNode*>(curr);
        if (!leaf->is_model_leafnode()) {
            if (!leaf->is_full()) {
                return leaf->insert_data(kv);
            }
            auto insert_pos = find_lower(leaf, kv.first);
            bool insert_left = insert_pos < (leaf->slotuse() >> 1);
            auto [left_leaf, right_leaf] = split_leaf_node(leaf);
            if (insert_left) {
                left_leaf->insert_data(kv);
            } else {
                right_leaf->insert_data(kv);
            }
            insert_new_node(right_leaf, static_cast<InnerNode*>(leaf->parent_node), true,
                            left_leaf->key(left_leaf->slotuse() - 1));
        }

        return true;
    }

    // void bulk_load(vector<kv_pair> &data_list) {
    //     std::sort(data_list.begin(), data_list.end(), [](const kv_pair& a, const kv_pair& b) {
    //         return a.first < b.first;
    //     });
    //     LeafNode* curr = new LeafNode();
    //     for (auto kv : data_list) {
    //         curr->insert_data(kv);
    //         if (curr->is_full()) {
    //
    //         }
    //     }
    // }

public:
    void update_leaves_chain(LeafNode* pre, LeafNode* curr, LeafNode* next) {
        if (pre) {
            pre->next_leaf = curr;
        } else {
            head_leaf = curr;
        }
        if (next) {
            next->prev_leaf = curr;
        } else {
            tail_leaf = curr;
        }
        curr->prev_leaf = pre;
        curr->next_leaf = next;
    }

    //! Split up an inner node into two equally-filled sibling nodes
    //! Return the (original_inner_node, new_inner_node)
    std::pair<InnerNode*, InnerNode*> split_inner_node(InnerNode* inner, key_type& split_key) {
        unsigned int mid = (inner->slotuse() >> 1);
        split_key = inner->slotkey[mid];

        InnerNode* newinner = new InnerNode(inner->level);

        // newinner->slotkey(std::make_move_iterator(inner->slotkey.begin() + mid + 1),
        //                   std::make_move_iterator(inner->slotkey.end()));
        std::copy(std::make_move_iterator(inner->slotkey.begin() + mid + 1),
                  std::make_move_iterator(inner->slotkey.end()), std::back_inserter(newinner->slotkey));
        // newinner->child_nodes(std::make_move_iterator(inner->child_nodes.begin() + mid + 1),
        //                   std::make_move_iterator(inner->child_nodes.end()));
        std::copy(std::make_move_iterator(inner->child_nodes.begin() + mid + 1),
                  std::make_move_iterator(inner->child_nodes.end()), std::back_inserter(newinner->child_nodes));

        // update the parent of the child nodes
        for (auto& child: newinner->child_nodes) {
            child->parent_node = newinner;
        }

        inner->slotkey.resize(mid);
        inner->child_nodes.resize(mid + 1);

        return std::make_pair(inner, newinner);
    }

    //! Split up a leaf node into two equally-filled sibling nodes
    //! Return the (original_leaf_node, new_leaf_node)
    std::pair<LeafNode*, LeafNode*> split_leaf_node(LeafNode* leaf, int split_pos = INT32_MIN) {
        split_pos = split_pos == INT32_MIN ? ((leaf->slotuse() >> 1) - 1) : split_pos;
        LeafNode* newleaf = new LeafNode();

        std::copy(std::make_move_iterator(leaf->slotdata.begin() + split_pos + 1),
                  std::make_move_iterator(leaf->slotdata.end()), std::back_inserter(newleaf->slotdata));

        leaf->slotdata.resize(split_pos + 1);

        return std::make_pair(leaf, newleaf);
    }

    // Insert a new node to its parent node
    // parent should not be null
    void insert_new_node(Node* new_node, InnerNode* parent, bool has_split_key, key_type pre_split_key) {
        if (new_node == nullptr) {
            fprintf(stderr, "Try to insert null node\n");
            abort();
        }

        if (!root) {
            root = new_node;
            head_leaf = static_cast<LeafNode*>(new_node);
            tail_leaf = static_cast<LeafNode*>(new_node);
            return;
        }

        if (!parent && has_split_key) {
            // Root node is leaf node and has splitted
            InnerNode* new_root = new InnerNode(root->level + 1);
            new_root->slotkey.emplace_back(pre_split_key);
            new_root->child_nodes.emplace_back(root);
            new_root->child_nodes.emplace_back(new_node);
            root->parent_node = new_root;
            new_node->parent_node = new_root;
            update_leaves_chain(static_cast<LeafNode*>(root), static_cast<LeafNode*>(new_node),
                                static_cast<LeafNode*>(root)->next_leaf);
            root = new_root;
            ++stats.num_inner_nodes;
            ++stats.num_inner_key_slots;
            stats.num_inner_ptr_slots += 2;
            return;
        }

        if (parent->is_full()) {
            int insert_pos = find_lower(parent, new_node->key(new_node->slotuse() - 1));
            bool is_left = insert_pos < FANOUT / 2;
            key_type split_key;
            auto [left, right] = split_inner_node(parent, split_key);

            // Update the key value of the left node in its parent node.
            if (left->parent_node) {
                ++stats.num_inner_ptr_slots;
                ++stats.num_inner_key_slots;
                ++stats.num_inner_nodes;
                insert_new_node(right, static_cast<InnerNode*>(left->parent_node), true, split_key);
            } else {
                // Root node is splitted
                InnerNode* new_root = new InnerNode(left->level + 1);
                new_root->slotkey.emplace_back(split_key);
                new_root->child_nodes.emplace_back(left);
                new_root->child_nodes.emplace_back(right);
                left->parent_node = new_root;
                right->parent_node = new_root;
                ++stats.num_inner_nodes;
                ++stats.num_inner_key_slots;
                stats.num_inner_ptr_slots += 2;
                root = new_root;
            }

            // Insert the new node to the parent node (left or right)
            if (is_left) {
                insert_new_node(new_node, left, has_split_key, pre_split_key);
            } else {
                insert_new_node(new_node, right, has_split_key, pre_split_key);
            }
        } else {
            int insert_pos;
            if (has_split_key) {
                insert_pos = find_lower(parent, pre_split_key);
                parent->slotkey.insert(parent->slotkey.begin() + insert_pos, pre_split_key);
                parent->child_nodes.insert(parent->child_nodes.begin() + insert_pos + 1, new_node);

                if (new_node->is_leafnode()) {
                    update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos]),
                                        static_cast<LeafNode*>(new_node),
                                        static_cast<LeafNode*>(parent->child_nodes[insert_pos])->next_leaf);
                }
            } else {
                insert_pos = find_lower(parent, new_node->key(new_node->slotuse() - 1));
                if (insert_pos == parent->slotuse() && new_node->key(0) > parent->child_nodes[insert_pos]->key(0)) {
                    // new node is the largest one
                    parent->slotkey.emplace_back(
                            parent->child_nodes[insert_pos]->key(parent->child_nodes[insert_pos]->slotuse() - 1));
                    parent->child_nodes.emplace_back(new_node);

                    update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos]),
                                        static_cast<LeafNode*>(new_node),
                                        static_cast<LeafNode*>(parent->child_nodes[insert_pos])->next_leaf);
                } else if (insert_pos == 0) {
                    unsigned int key_index = parent->child_nodes[insert_pos]->slotuse() > 0
                                                     ? parent->child_nodes[insert_pos]->slotuse() - 1
                                                     : 0;
                    if (parent->child_nodes[insert_pos]->key(key_index) < new_node->key(0)) {
                        parent->slotkey.insert(parent->slotkey.begin() + insert_pos,
                                               parent->child_nodes[insert_pos]->key(key_index));
                        parent->child_nodes.insert(parent->child_nodes.begin() + insert_pos + 1, new_node);

                        update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos]),
                                            static_cast<LeafNode*>(new_node),
                                            static_cast<LeafNode*>(parent->child_nodes[insert_pos])->next_leaf);
                    } else {
                        parent->slotkey.insert(parent->slotkey.begin() + insert_pos,
                                               new_node->key(new_node->slotuse() - 1));
                        parent->child_nodes.insert(parent->child_nodes.begin() + insert_pos, new_node);

                        update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos + 1])->prev_leaf,
                                            static_cast<LeafNode*>(new_node),
                                            static_cast<LeafNode*>(parent->child_nodes[insert_pos + 1]));
                    }
                } else {
                    parent->slotkey.insert(parent->slotkey.begin() + insert_pos,
                                           new_node->key(new_node->slotuse() - 1));
                    parent->child_nodes.insert(parent->child_nodes.begin() + insert_pos, new_node);

                    update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos + 1])->prev_leaf,
                                        static_cast<LeafNode*>(new_node),
                                        static_cast<LeafNode*>(parent->child_nodes[insert_pos + 1]));
                }
            }
            ++stats.num_inner_ptr_slots;
            ++stats.num_inner_key_slots;
            new_node->parent_node = parent;
        }
    }

    // Delete one item from the tree
    bool erase(key_type& original_key) {
        key_type key = original_key << 1;

        if (!root)
            return false;
        LeafNode* leaf = find_leaf(key);

        if (!leaf)
            return false;

        if (!leaf->is_model_leafnode()) {
            leaf->erase_data(key);
            if (leaf->is_underflow() && leaf != root) {
                auto [left_leaf, right_leaf] = find_siblings(leaf);
                bool right_mergeable = right_leaf && right_leaf->is_few();
                bool left_mergeable = left_leaf && left_leaf->is_few();
                bool is_choose_left = (left_mergeable && !right_mergeable) ||
                                      (left_leaf && left_leaf->parent_node == leaf->parent_node &&
                                       ((!left_mergeable && !right_mergeable) || (left_mergeable && right_mergeable)));
                LeafNode* node_prime = nullptr;
                if (is_choose_left && !left_leaf->is_model_leafnode()) {
                    node_prime = static_cast<LeafNode*>(left_leaf);
                } else {
                    node_prime = static_cast<LeafNode*>(right_leaf);
                }

                if (!node_prime->is_model_leafnode()) {
                    if (node_prime->slotuse() + leaf->slotuse() <= leaf_slotmax) {
                        if (is_choose_left)
                            merge_leaf(node_prime, leaf, node_prime->slotuse() - 1);
                        else
                            merge_leaf(leaf, node_prime, leaf->slotuse() - 1);
                    } else {
                        if (node_prime->parent_node != leaf->parent_node) {
                            if (is_choose_left) {
                                adjust_parent_nodes(static_cast<InnerNode*>(node_prime->parent_node),
                                                    static_cast<InnerNode*>(leaf->parent_node));
                            } else {
                                adjust_parent_nodes(static_cast<InnerNode*>(leaf->parent_node),
                                                    static_cast<InnerNode*>(node_prime->parent_node));
                            }
                        }

                        if (is_choose_left) {
                            int parent_slot = find_lower(node_prime->parent_node, node_prime->key(0));
                            shift_right_leaf(node_prime, leaf, static_cast<InnerNode*>(node_prime->parent_node),
                                             parent_slot);
                        } else {
                            int parent_slot = find_lower(leaf->parent_node, leaf->key(0));
                            shift_left_leaf(leaf, node_prime, static_cast<InnerNode*>(node_prime->parent_node),
                                            parent_slot);
                        }
                    }
                }
            }
        } else {
            auto* learned_leaf = static_cast<LearnedLeafNode*>(leaf);
            int slot = learned_leaf->search_model(key);
            if (learned_leaf->slotdata[slot] == key) {
                learned_leaf->slotdata[slot] |= 1;
            } else {
                auto it = std::find(learned_leaf->buffer_slotdata.begin(), learned_leaf->buffer_slotdata.end(), key);
                learned_leaf->buffer_slotdata.erase(it);
            }
        }

        return true;
    }

    //! Merge two inner nodes. The function moves all key/childid pairs from
    //! right to left and sets right's slotuse to zero. The right slot is then
    //! removed by the calling parent node.
    void merge_inner(InnerNode* left, InnerNode* right, InnerNode* parent, unsigned int parentslot) {
        // retrieve the decision key from parent
        left->slotkey.emplace_back(parent->slotkey[parentslot]);
        auto pre_slotuse = left->slotuse();
        auto first_r = *right->slotkey.begin();

        left->slotkey.insert(left->slotkey.end(), right->slotkey.begin(), right->slotkey.end());
        left->child_nodes.insert(left->child_nodes.end(), right->child_nodes.begin(), right->child_nodes.end());

        // update parent node pointer in children
        for (unsigned int i = 0; i <= right->slotuse(); ++i) {
            left->child_nodes[pre_slotuse + i]->parent_node = left;
        }

        // right->slotkey.clear();
        // right->slotkey.emplace_back(first_r);
    }

    //! Balance two inner nodes. The function moves key/data pairs from left to
    //! right so that both nodes are equally filled. The parent node is updated
    //! if possible.
    void shift_right_inner(InnerNode* left, InnerNode* right, InnerNode* parent, unsigned int parentslot) {
        unsigned int shiftnum =
                (left->slotuse() - right->slotuse()) >> 1 > 0 ? (left->slotuse() - right->slotuse()) >> 1 : 1;

        right->slotkey.insert(right->slotkey.begin(), parent->slotkey[parentslot]);
        ;

        // copy the remaining last items from the left node to the first slot in
        // the right node.
        right->slotkey.insert(right->slotkey.begin(), left->slotkey.end() - shiftnum + 1, left->slotkey.end());
        right->child_nodes.insert(right->child_nodes.begin(), left->child_nodes.end() - shiftnum,
                                  left->child_nodes.end());

        // update parent pointer shift from left to right
        for (unsigned int i = 0; i < shiftnum; ++i) {
            right->child_nodes[i]->parent_node = right;
        }

        // copy the first to-be-removed key from the left node to the parent's
        // decision slot
        parent->slotkey[parentslot] = left->slotkey[left->slotuse() - shiftnum];

        left->slotkey.resize(left->slotkey.size() - shiftnum);
        left->child_nodes.resize(left->child_nodes.size() - shiftnum);
    }

    //! Balance two inner nodes. The function moves key/data pairs from right to
    //! left so that both nodes are equally filled. The parent node is updated
    //! if possible
    void shift_left_inner(InnerNode* left, InnerNode* right, InnerNode* parent, unsigned int parentslot) {
        unsigned int shiftnum =
                (right->slotuse() - left->slotuse()) >> 1 > 0 ? (right->slotuse() - left->slotuse()) >> 1 : 1;

        // copy the parent's decision slotkey and childid to the first new key
        // on the left
        left->slotkey.emplace_back(parent->slotkey[parentslot]);

        auto pre_slotuse = left->slotuse();

        // copy the other items from the right node to the last slots in the
        // left node.
        left->slotkey.insert(left->slotkey.end(), right->slotkey.begin(), right->slotkey.begin() + shiftnum - 1);
        left->child_nodes.insert(left->child_nodes.end(), right->child_nodes.begin(),
                                 right->child_nodes.begin() + shiftnum);

        // update parent pointer
        for (unsigned int i = 0; i < shiftnum; ++i) {
            left->child_nodes[pre_slotuse + i]->parent_node = left;
        }

        // fixup parent
        parent->slotkey[parentslot] = right->slotkey[shiftnum - 1];

        // shift all slots in the right node
        std::move(right->slotkey.begin() + shiftnum, right->slotkey.end(), right->slotkey.begin());
        std::move(right->child_nodes.begin() + shiftnum, right->child_nodes.end(), right->child_nodes.begin());
        right->slotkey.resize(right->slotkey.size() - shiftnum);
        right->child_nodes.resize(right->child_nodes.size() - shiftnum);
    }

    //! Balance two leaf nodes. The function moves key/data pairs from left to
    //! right so that both nodes are equally filled. The parent node is updated
    //! if possible.
    void shift_right_leaf(LeafNode* left, LeafNode* right, InnerNode* parent, unsigned int parentslot) {
        unsigned int shiftnum = (left->slotuse() - right->slotuse()) >> 1;

        // copy the last items from the left node to the first slot in the right
        // node.
        right->slotdata.insert(right->slotdata.begin(), left->slotdata.end() - shiftnum, left->slotdata.end());
        left->slotdata.resize(left->slotdata.size() - shiftnum);

        parent->slotkey[parentslot] = left->key(left->slotuse() - 1);
    }

    //! Balance two leaf nodes. The function moves key/data pairs from right to
    //! left so that both nodes are equally filled. The parent node is updated
    //! if possible.
    void shift_left_leaf(LeafNode* left, LeafNode* right, InnerNode* parent, unsigned int parentslot) {
        unsigned int shiftnum = (right->slotuse() - left->slotuse()) >> 1;

        // copy the first items from the right node to the last slot in the left
        // node.
        left->slotdata.insert(left->slotdata.end(), right->slotdata.begin(), right->slotdata.begin() + shiftnum);

        // shift all slots in the right node
        std::move(right->slotdata.begin() + shiftnum, right->slotdata.end(), right->slotdata.begin());
        right->slotdata.resize(right->slotdata.size() - shiftnum);

        parent->slotkey[parentslot] = left->key(left->slotuse() - 1);
    }


    void merge_leaf(LeafNode* left_node, LeafNode* right_node, unsigned int merge_key_idx, bool verbose = false,
                    std::ostream* os = &std::cout) {
        InnerNode* left_parent = static_cast<InnerNode*>(left_node->parent_node);
        InnerNode* right_parent = static_cast<InnerNode*>(right_node->parent_node);

        if (left_parent == right_parent) {
            // Two leaf nodes are sharing same parent node
            key_type merge_key = left_parent->key(find_lower(left_parent, left_node->key(merge_key_idx)));
            left_node->slotdata.insert(left_node->slotdata.end(), right_node->slotdata.begin(),
                                       right_node->slotdata.end());
            delete_node(right_node, true, merge_key);
        } else {
#ifndef NODEBUG
            bool is_check = check_structure();
#endif
            // Make left and right parents become same
            adjust_parent_nodes(left_parent, right_parent, verbose, os);
#ifndef NODEBUG
            left_parent = static_cast<InnerNode*>(left_node->parent_node);
            right_parent = static_cast<InnerNode*>(right_node->parent_node);
            check_structure();
            if (left_parent != right_parent) {
                // fmt::println("Still left and right parent are not same");
                std::cout << "Still left and right parent are not same" << std::endl;
                abort();
            }
#endif
            merge_leaf(left_node, right_node, merge_key_idx, verbose, os);
        }
    }

    // Make the two inner nodes left and right to have the same parent node
    void adjust_parent_nodes(InnerNode* left_parent, InnerNode* right_parent, bool verbose = false,
                             std::ostream* os = &std::cout) {
        unsigned int k_prime_pos = find_lower(left_parent->parent_node, left_parent->key(left_parent->slotuse() - 1));
        if (left_parent->slotuse() + right_parent->slotuse() < inner_slotmax) {
            if (left_parent->parent_node != right_parent->parent_node) {
                adjust_parent_nodes(static_cast<InnerNode*>(left_parent->parent_node),
                                    static_cast<InnerNode*>(right_parent->parent_node), verbose, os);
                k_prime_pos = find_lower(left_parent->parent_node, left_parent->key(left_parent->slotuse() - 1));
#ifndef NODEBUG
                if (verbose) {
                    print_friendly(*os);
                }
#endif
            }
            // Merge two parents node
            merge_inner(left_parent, right_parent, static_cast<InnerNode*>(left_parent->parent_node), k_prime_pos);
            delete_node(right_parent, true, static_cast<InnerNode*>(left_parent->parent_node)->key(k_prime_pos));
        } else {
            // Redistribute keys of left and right parents
            if (left_parent->parent_node != right_parent->parent_node) {
                adjust_parent_nodes(static_cast<InnerNode*>(left_parent->parent_node),
                                    static_cast<InnerNode*>(right_parent->parent_node), verbose, os);
                k_prime_pos = find_lower(left_parent->parent_node, left_parent->key(left_parent->slotuse() - 1));
#ifndef NODEBUG
                if (verbose) {
                    print_friendly(*os);
                }
#endif
            }
            if (left_parent->slotuse() > right_parent->slotuse()) {
                shift_right_inner(left_parent, right_parent, static_cast<InnerNode*>(left_parent->parent_node),
                                  k_prime_pos);
            } else {
                shift_left_inner(left_parent, right_parent, static_cast<InnerNode*>(left_parent->parent_node),
                                 k_prime_pos);
            }
        }
    }

    // Delete one node from the tree
    bool delete_node(Node* del_node, bool has_merged, key_type push_down_key) {
        if (!del_node) {
            fprintf(stderr, "Try to delete null node\n");
            abort();
        }

        InnerNode* parent = static_cast<InnerNode*>(del_node->parent_node);
        if (!parent) {
            del_node = nullptr;
            free_node(del_node);
            root = nullptr;
            return true;
        }

        auto siblings = find_siblings(parent);
        InnerNode* left_parent = static_cast<InnerNode*>(siblings.first);
        InnerNode* right_parent = static_cast<InnerNode*>(siblings.second);

        if (del_node->is_leafnode()) {
            LeafNode* leaf = static_cast<LeafNode*>(del_node);
            if (leaf->prev_leaf == nullptr || leaf->next_leaf == nullptr) {
                if (leaf->prev_leaf) {
                    tail_leaf = leaf->prev_leaf;
                    tail_leaf->next_leaf = nullptr;
                } else {
                    head_leaf = leaf->next_leaf;
                    head_leaf->prev_leaf = nullptr;
                }
            } else {
                leaf->prev_leaf->next_leaf = leaf->next_leaf;
                leaf->next_leaf->prev_leaf = leaf->prev_leaf;
            }
        }
        if (!has_merged) {
            key_type del_key = del_node->key(del_node->slotuse() > 0 ? del_node->slotuse() - 1 : 0);
            unsigned int del_pos = find_lower(parent, del_key);
            free_node(del_node);
            del_node = nullptr;

            parent->slotkey.erase(parent->slotkey.begin() + del_pos);
            parent->child_nodes.erase(parent->child_nodes.begin() + del_pos);
        } else {
            unsigned int del_pos = find_lower(parent, push_down_key);
            free_node(del_node);
            del_node = nullptr;

            parent->slotkey.erase(parent->slotkey.begin() + del_pos);
            parent->child_nodes.erase(parent->child_nodes.begin() + del_pos + 1);
        }

        // Root parent delete node
        if (root == parent) {
            if (root->slotuse() == 0) {
                root = parent->child_nodes[0];
                free_node(parent);
                root->parent_node = nullptr;
            }
            return true;
        }

        // Non root parent delete node
        if (parent->is_underflow()) {
            bool is_choose_left = right_parent == nullptr ||
                                  (left_parent != nullptr && left_parent->slotuse() < right_parent->slotuse());
            bool is_only_one_neighbor = left_parent == nullptr || right_parent == nullptr;
            InnerNode* node_prime = nullptr;
            unsigned int k_prime_pos;
            if (is_choose_left) {
                node_prime = static_cast<InnerNode*>(left_parent);
                k_prime_pos = find_lower(left_parent->parent_node, left_parent->key(left_parent->slotuse() - 1));
            } else {
                node_prime = static_cast<InnerNode*>(right_parent);
                k_prime_pos =
                        find_lower(parent->parent_node, parent->key(parent->slotuse() > 0 ? parent->slotuse() - 1 : 0));
            }

            // Can merge parent and its neighborhood
            if (node_prime->slotuse() + parent->slotuse() < inner_slotmax) {
                if (!is_choose_left) {
                    std::swap(parent, node_prime);
                }
                merge_inner(node_prime, parent, static_cast<InnerNode*>(node_prime->parent_node), k_prime_pos);
                if (node_prime->parent_node == nullptr) {
                    fprintf(stderr, "node_prime's parent node is null\n");
                }

                delete_node(parent, true, static_cast<InnerNode*>(node_prime->parent_node)->slotkey[k_prime_pos]);
                parent = nullptr;
            } else {
                // Borrow from parent's neighborhood
                if (is_only_one_neighbor) {
                    if (is_choose_left) {
                        shift_right_inner(node_prime, parent, static_cast<InnerNode*>(node_prime->parent_node),
                                          k_prime_pos);
                    } else {
                        shift_left_inner(parent, node_prime, static_cast<InnerNode*>(node_prime->parent_node),
                                         k_prime_pos);
                    }
                } else {
                    if (is_choose_left) {
                        k_prime_pos = find_lower(parent->parent_node, parent->key(parent->slotuse() - 1));
                        shift_left_inner(parent, right_parent, static_cast<InnerNode*>(parent->parent_node),
                                         k_prime_pos);
                    } else {
                        k_prime_pos =
                                find_lower(left_parent->parent_node, left_parent->key(left_parent->slotuse() - 1));
                        shift_right_inner(left_parent, parent, static_cast<InnerNode*>(left_parent->parent_node),
                                          k_prime_pos);
                    }
                }
            }
        }
        return true;
    }

    // Find siblings of a node
    std::pair<Node*, Node*> find_siblings(Node* node) {
        if (node->is_leafnode()) {
            return std::make_pair(static_cast<LeafNode*>(node)->prev_leaf, static_cast<LeafNode*>(node)->next_leaf);
        }

        if (node->parent_node == nullptr) {
            return std::make_pair(nullptr, nullptr);
        }


        unsigned int parent_slot = find_lower(node->parent_node, node->key(node->slotuse() - 1));
        if (parent_slot == 0) {
            return std::make_pair(nullptr, static_cast<InnerNode*>(node->parent_node)->child_nodes[1]);
        } else if (parent_slot == node->parent_node->slotuse()) {
            return std::make_pair(static_cast<InnerNode*>(node->parent_node)->child_nodes[parent_slot - 1], nullptr);
        } else {
            return std::make_pair(static_cast<InnerNode*>(node->parent_node)->child_nodes[parent_slot - 1],
                                  static_cast<InnerNode*>(node->parent_node)->child_nodes[parent_slot + 1]);
        }
    }

private:
    void merge_model_leaf_with_duplicate_leaf(bool& is_linked_to_tree, LeafNode* model_leaf, LeafNode* dup_leaf,
                                              bool verbose = false, std::ostream* os = &std::cout) {
        if (!is_linked_to_tree) {
            auto pre = dup_leaf->prev_leaf;
            update_leaves_chain(pre, model_leaf, dup_leaf->next_leaf);
            model_leaf->parent_node = dup_leaf->parent_node;
            unsigned int ins_idx = find_lower(dup_leaf->parent_node, dup_leaf->key(dup_leaf->slotuse() - 1));
            static_cast<InnerNode*>(dup_leaf->parent_node)->child_nodes[ins_idx] = model_leaf;
            free_node(dup_leaf);
            is_linked_to_tree = true;
#ifndef NODEBUG
            if (!this->check_structure(*os)) {
                // fmt::println("Wrong structure");
                std::cout << "Wrong structure" << std::endl;
                print_friendly(*os);
                abort();
            }
            if (verbose) {
                print_friendly(*os);
            }
#endif
        } else {
            InnerNode* left_parent = static_cast<InnerNode*>(model_leaf->parent_node);
            InnerNode* right_parent = static_cast<InnerNode*>(dup_leaf->parent_node);
            if (left_parent == right_parent) {
                key_type merge_key = model_leaf->key(model_leaf->slotuse() - dup_leaf->slotuse() - 1);
                delete_node(dup_leaf, true, merge_key);
            } else {
                adjust_parent_nodes(left_parent, right_parent, verbose, os);
#ifndef NODEBUG
                if (verbose) {
                    print_friendly(*os);
                }
                left_parent = static_cast<InnerNode*>(model_leaf->parent_node);
                right_parent = static_cast<InnerNode*>(dup_leaf->parent_node);
                if (left_parent != right_parent) {
                    // fmt::println("Still left and right parent are not same");
                    std::cout << "Still left and right parent are not same" << std::endl;
                    abort();
                }
#endif
                key_type merge_key = model_leaf->key(model_leaf->slotuse() - dup_leaf->slotuse() - 1);
                delete_node(dup_leaf, true, merge_key);
            }
#ifndef NODEBUG
            if (!this->check_structure(*os)) {
                // fmt::println("Wrong structure");
                std::cout << "Wrong structure" << std::endl;
                print_friendly(*os);
                abort();
            }
            if (verbose) {
                print_friendly(*os);
            }
#endif
        }
    }

    // Return next curr node ptr
    LeafNode* add_model_leaf(LearnedLeafNode* model_leaf, LeafNode* curr, int curr_num_added, bool& is_linked_to_tree,
                             bool is_last, bool verbose = false, std::ostream* os = &std::cout) {
        if (is_linked_to_tree && curr_num_added == 0) return curr;

        if (is_last) {
            // Current node is the last node
            // Abort adding last leaf node data to the model leaf
            if (is_linked_to_tree) {
                model_leaf->slotdata.resize(model_leaf->slotuse() - curr_num_added);
                model_leaf->retrain();
            } else {
                free_node(model_leaf);
            }

            return curr;
        }

        auto [left_curr, right_curr] = split_leaf_node(curr, curr_num_added - 1);
        key_type split_key = left_curr->key(left_curr->slotuse() - 1);

        // Merge model leaf with left_curr as left_curr's slotdata have already been added to the model_leaf
        merge_model_leaf_with_duplicate_leaf(is_linked_to_tree, model_leaf, left_curr, verbose, os);

        if (right_curr->slotuse() != 0) {
            insert_new_node(right_curr, static_cast<InnerNode*>(model_leaf->parent_node), true, split_key);

#ifndef NODEBUG
            if (!this->check_structure(*os)) {
                // fmt::println("Wrong structure");
                std::cout << "Wrong structure" << std::endl;
                print_friendly(*os);
                abort();
            }
            if (verbose) {
                print_friendly(*os);
            }
#endif


            if (right_curr->is_underflow()) {
                // Merge right_curr with right_curr->next_leaf
                if (right_curr->slotuse() + right_curr->next_leaf->slotuse() <= leaf_slotmax) {
                    merge_leaf(right_curr, right_curr->next_leaf, right_curr->slotuse() - 1, verbose, os);

#ifndef NODEBUG
                    if (verbose) {
                        print_friendly(*os);
                    }
#endif

                } else {
                    // Rebalance right_curr with right_curr->next_leaf
                    if (right_curr->parent_node == right_curr->next_leaf->parent_node) {
                        // Two nodes have the same parent
                        unsigned int parent_slot =
                                find_lower(static_cast<InnerNode*>(right_curr->parent_node), right_curr->key(0));
                        shift_left_leaf(right_curr, right_curr->next_leaf,
                                        static_cast<InnerNode*>(right_curr->parent_node), parent_slot);
                    } else {
                        // Two nodes have different parent
                        adjust_parent_nodes(static_cast<InnerNode*>(right_curr->parent_node),
                                            static_cast<InnerNode*>(right_curr->next_leaf->parent_node), verbose, os);

#ifndef NODEBUG
                        if (right_curr->parent_node != right_curr->next_leaf->parent_node) {
                            // fmt::println("Still left and right parent are not same");
                            std::cout << "Still left and right parent are not same" << std::endl;
                            abort();
                        }
#endif

                        unsigned int parent_slot =
                                find_lower(static_cast<InnerNode*>(right_curr->parent_node), right_curr->key(0));
                        shift_left_leaf(right_curr, right_curr->next_leaf,
                                        static_cast<InnerNode*>(right_curr->parent_node), parent_slot);
                    }
                }
            }
        }

        return model_leaf->next_leaf;
    }

public:
    void optimize_leaves_with_model_v2(LeafNode* start, LeafNode* end, bool verbose = false,
                                       std::ostream* os = &std::cout) {
        if (!root) return;

        LeafNode* curr = start;
        LearnedLeafNode* model_leaf = new LearnedLeafNode();

        int curr_cnt = 0;
        bool is_finished = false;
        bool is_added_to_tree = false;
        LeafNode* next = nullptr;

        while (!is_finished) {
            if (curr->next_leaf == end) is_finished = true;
            next = nullptr;
            for (int i = 0; i < curr->slotuse(); ++i) {
                bool is_added = model_leaf->model->add_point(curr->key(i), model_leaf->slotuse());
                if (is_added) {
                    model_leaf->slotdata.emplace_back(curr->slotdata[i]);
                    ++curr_cnt;
                    if (model_leaf->is_full()) {
                        model_leaf->get_model_param();
                        next = add_model_leaf(model_leaf, curr, curr_cnt, is_added_to_tree, is_finished, verbose, os);
                        if (!is_finished) model_leaf = new LearnedLeafNode();
#ifndef NODEBUG
                        if (!this->check_structure(*os)) {
                            // fmt::println("Wrong structure");
                            std::cout << "Wrong structure" << std::endl;
                            print_friendly(*os);
                            abort();
                        }
                        if (verbose) {
                            print_friendly(*os);
                        }
#endif
                        break;
                    }
                } else {
                    if (model_leaf->is_underflow()) {
                        free_node(model_leaf);
                        model_leaf = nullptr;
                        if (!is_finished) model_leaf = new LearnedLeafNode();
                        next = curr->next_leaf;
                        break;
                    } else {
                        model_leaf->get_model_param();
                        next = add_model_leaf(model_leaf, curr, curr_cnt, is_added_to_tree, is_finished, verbose, os);
                        if (!is_finished) model_leaf = new LearnedLeafNode();
#ifndef NODEBUG
                        if (!this->check_structure(*os)) {
                            // fmt::println("Wrong structure");
                            std::cout << "Wrong structure" << std::endl;
                            print_friendly(*os);
                            abort();
                        }
                        if (verbose) {
                            print_friendly(*os);
                        }
#endif
                        break;
                    }
                }
            }

            if (next == nullptr && model_leaf) {
                // Current node data has been added to the model leaf
                next = curr->next_leaf;
                merge_model_leaf_with_duplicate_leaf(is_added_to_tree, model_leaf, curr, verbose, os);
                curr = next;
                curr_cnt = 0;
            } else {
                // Prev model leaf has been added to the tree or deleted
                is_added_to_tree = false;
                curr_cnt = 0;
                curr = next;
#ifndef NODEBUG
                if (!this->check_structure(*os)) {
                    // fmt::println("Wrong structure");
                    std::cout << "Wrong structure" << std::endl;
                }
                if (verbose) {
                    print_friendly(*os);
                }
#endif
            }
        }
    }

public:
    // Print tree structure
    void print_friendly(std::ostream& os = std::cout) {
        if (root == nullptr) return;

        os << "------------" << std::endl;
        std::queue<Node*> queue;
        queue.push(root);

        int n_level = 1;  // Number of nodes in current level
        int n_nextLevel = 0;
        int curr_level = root->level;

        while (queue.empty() == false) {
            Node* node = queue.front();
            if (node->level != curr_level) {
                curr_level = node->level;
                os << "====" << std::endl;
                os.flush();
            }

            os << "|";
            os.flush();
            if (node->is_leafnode()) {
                bool is_model_leaf = node->is_model_leafnode();
                if (is_model_leaf) {
                    // Merge slotdata and buffer data
                    LearnedLeafNode* leaf = static_cast<LearnedLeafNode*>(node);
                    std::vector<key_type> merged_data(leaf->slotdata.size() + leaf->buffer_slotdata.size());
                    std::sort(leaf->buffer_slotdata.begin(), leaf->buffer_slotdata.end());
                    std::merge(leaf->slotdata.begin(), leaf->slotdata.end(), leaf->buffer_slotdata.begin(),
                               leaf->buffer_slotdata.end(), merged_data.begin());
                     for (unsigned int i = 0; i < merged_data.size(); i++) {
                        os << (merged_data[i] >> 1) << "|";
                        os.flush();
                    }
                } else {
                    LeafNode* leaf = static_cast<LeafNode*>(node);
                    for (unsigned int i = 0; i < leaf->slotuse(); i++) {
                        os << (leaf->key(i) >> 1) << "|";
                        os.flush();
                    }
                }
            } else {
                InnerNode* inner = static_cast<InnerNode*>(node);
                for (unsigned int i = 0; i < inner->slotuse(); i++) {
                    os << (inner->slotkey[i] >> 1) << "|";
                    os.flush();
                    queue.push(inner->child_nodes[i]);
                    n_nextLevel++;
                }
                queue.push(inner->child_nodes[inner->slotuse()]);
                n_nextLevel++;
            }

            queue.pop();
            n_level--;
            os << "   ";
            os.flush();

            if (n_level == 0) {
                os << std::endl;
                os.flush();
                n_level = n_nextLevel;
                n_nextLevel = 0;
            }
        }
        // std::cout << std::flush;
        os << "------------" << std::endl;
    }

private:
    // Check if the parent node is correct
    bool check_parent_node(std::ostream& os = std::cout) {
        bool is_correct = true;

        if (root == nullptr) return true;

        if (root->parent_node != nullptr) {
            os << "Root node has parent node" << std::endl;
            os.flush();
            is_correct = false;
            return is_correct;
        }

        std::queue<Node*> queue;
        queue.push(root);

        while (queue.empty() == false) {
            Node* node = queue.front();

            if (!node->is_leafnode()) {
                InnerNode* inner = static_cast<InnerNode*>(node);
                for (unsigned int i = 0; i < inner->slotuse(); i++) {
                    if (inner->child_nodes[i]->parent_node != inner) {
                        os << "Parent node is not correct" << std::endl;
                        is_correct = false;
                        return is_correct;
                    }
                    queue.push(inner->child_nodes[i]);
                }
                if (inner->child_nodes[inner->slotuse()]->parent_node != inner) {
                    os << "Parent node is not correct" << std::endl;
                    is_correct = false;
                    return is_correct;
                }
                queue.push(inner->child_nodes[inner->slotuse()]);
            }

            queue.pop();
        }

        if (!is_correct) {
            os << "Parent node is not correct" << std::endl;
        }
        return is_correct;
    }

private:
public:
    // Check if the structure of the b+-tree is correct
    bool check_structure(std::ostream& os = std::cout) {
        if (root == nullptr) {
            return true;
        } else {
            bool is_parent_correct = check_parent_node(os);
            bool is_node_correct = check_node(root, 0, ((key_type)0 - (key_type)1), os);
            bool is_leaf_correct = check_leaves_link(os);
            return is_leaf_correct && is_node_correct && is_parent_correct;
        }
    }

private:
    bool check_node(Node* node, key_type min_bound, key_type max_bound, std::ostream& os = std::cout) {
        if (!node) return true;
        // if (node != root && node->is_underflow()) {
        //     os << "Node is underflow" << std::endl;
        //     return false;
        // }

        // check key order
        if (!node->is_leafnode()) {
            InnerNode* inner_node = static_cast<InnerNode*>(node);
            for (int i = 0; i < inner_node->slotuse() - 1; i++) {
                if (inner_node->slotkey[i + 1] < inner_node->slotkey[i]) {
                    os << "Node keys are not sorted" << std::endl;
                    return false;
                }

                if (inner_node->slotkey[i] > max_bound) {
                    os << "Prev node max bound key is less than node key" << std::endl;
                    return false;
                }

                // if (min_bound >= inner_node->slotkey[i]) {
                //     os << "Prev node min bound key is greater than node key" << std::endl;
                //     return false;
                // }
            }
        } else {
            LeafNode* leaf_node = static_cast<LeafNode*>(node);
            for (int i = 0; i < (int) leaf_node->slotuse() - 1; i++) {
                if (leaf_node->key(i + 1) < leaf_node->key(i)) {
                    os << "Node keys are not sorted" << std::endl;
                    return false;
                }

                if (leaf_node->key(i) > max_bound) {
                    os << "Prev node bound key is less than node key" << std::endl;
                    return false;
                }

                // if (min_bound >= leaf_node->key(i)) {
                //     os << "Prev node min bound key is greater than node key" << std::endl;
                //     return false;
                // }
            }
        }

        // Check parent and child key relationship
        if (!node->is_leafnode()) {
            InnerNode* inner_node = static_cast<InnerNode*>(node);
            for (int i = 0; i < inner_node->slotuse(); i++) {
                key_type parent_key = inner_node->key(i);
                if (i == 0) {
                    if (check_node(inner_node->child_nodes[i], min_bound, parent_key, os) == false) {
                        return false;
                    }
                } else {
                    if (check_node(inner_node->child_nodes[i], inner_node->key(i - 1), parent_key, os) == false) {
                        return false;
                    }
                }

                for (int j = 0; j < inner_node->child_nodes[i]->slotuse(); j++) {
                    if (parent_key < inner_node->child_nodes[i]->key(j)) {
                        os << "Parent key is less than child key" << std::endl;
                        return false;
                    }
                }
            }
            if (check_node(inner_node->child_nodes[inner_node->slotuse()], inner_node->key(inner_node->slotuse() - 1),
                           max_bound, os) == false) {
                return false;
            }
            for (int j = 0; j < inner_node->child_nodes[inner_node->slotuse() - 1]->slotuse(); j++) {
                if (inner_node->child_nodes[inner_node->slotuse()]->key(0) <
                    inner_node->key(inner_node->slotuse() - 1)) {
                    os << "Last parent key is greater than its last child key" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    bool check_leaves_link(std::ostream& os = std::cout) {
        Node* tmp = root;
        while (tmp && tmp->is_leafnode() == false) tmp = static_cast<InnerNode*>(tmp)->child_nodes[0];

        if (head_leaf != tmp) {
            os << "Wrong head leaf" << std::endl;
            return false;
        }

        std::queue<LeafNode*> leaf_queue;
        std::queue<Node*> queue;
        queue.push(root);

        while (queue.empty() == false) {
            Node* node = queue.front();
            if (!node->is_leafnode()) {
                InnerNode* inner = static_cast<InnerNode*>(node);
                for (unsigned int i = 0; i <= inner->slotuse(); i++) {
                    queue.push(inner->child_nodes[i]);
                }
            } else {
                LeafNode* leaf = static_cast<LeafNode*>(node);
                leaf_queue.push(leaf);
            }
            queue.pop();
        }

        LeafNode* leaf_node = static_cast<LeafNode*>(tmp);
        while (leaf_node && leaf_node->next_leaf) {
            LeafNode* queue_leaf = leaf_queue.front();
            if (queue_leaf != leaf_node) {
                os << "Leaf nodes are not linked correctly (Next ptr)" << std::endl;
                return false;
            }
            leaf_queue.pop();
            if (leaf_node->next_leaf->key(0) < leaf_node->key(leaf_node->slotuse() - 1)) {
                os << "Leaf nodes are not linked correctly (Pre ptr)" << std::endl;
                return false;
            }

            if (leaf_node->next_leaf->prev_leaf != leaf_node) {
                os << "Previous leaf node is not linked correctly" << std::endl;
                return false;
            }

            leaf_node = leaf_node->next_leaf;
        }

        if (tail_leaf != leaf_node) {
            os << "Wrong tail leaf" << std::endl;
            return false;
        }
        return true;
    }

public:
    // Some statistics
    std::pair<size_t, size_t> mem_usage() {
        size_t inner_mem = 0;
        size_t leaf_mem = 0;

        if (!root) return {0, 0};

        Node* curr = root;
        std::queue<Node*> queue;
        queue.push(curr);

        while (!queue.empty()) {
            Node* node = queue.front();
            if (node->is_leafnode())
                leaf_mem += node->mem_size();
            else {
                inner_mem += node->mem_size();
                InnerNode* inner = static_cast<InnerNode*>(node);
                for (unsigned int i = 0; i <= inner->slotuse(); i++) {
                    queue.push(inner->child_nodes[i]);
                }
            }
            queue.pop();
        }
        return std::make_pair(inner_mem, leaf_mem);
    }

    double average_leaf_load() {
        size_t leaf_cnt = 0;
        size_t total_load = 0;
        LeafNode* curr = head_leaf;

        while (curr != nullptr) {
            total_load += curr->slotuse();
            ++leaf_cnt;
            curr = curr->next_leaf;
        }

        return (double) total_load / (double) leaf_cnt;
    }

    // <<model_leaf_num, total_leaves_num>, <model_data_num, total_data_num>>
    std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>> learned_leaf_ratio() {
        size_t model_leaf_num = 0;
        size_t total_leaves_num = 0;

        size_t model_data_num = 0;
        size_t total_data_num = 0;
        LeafNode* curr = head_leaf;

        while (curr != nullptr) {
            if (curr->is_model_leafnode()) {
                model_data_num += curr->slotuse();
                ++model_leaf_num;
            }
            total_data_num += curr->slotuse();
            ++total_leaves_num;
            curr = curr->next_leaf;
        }

        auto model_leaf_ratio = std::make_pair(model_leaf_num, total_leaves_num);
        auto model_data_ratio = std::make_pair(model_data_num, total_data_num);
        return std::make_pair(model_leaf_ratio, model_data_ratio);
    }

    size_t size() {
        size_t total_load = 0;
        LeafNode* curr = head_leaf;

        while (curr != nullptr) {
            total_load += curr->slotuse();
            curr = curr->next_leaf;
        }

        return total_load;
    }

private:
    Node* copy_node(Node* node) {
        if (!node) return nullptr;

        Node* new_node = nullptr;

        if (node->is_model_leafnode()) {
            new_node = new LearnedLeafNode();
            LearnedLeafNode* new_leaf = static_cast<LearnedLeafNode*>(new_node);
            LearnedLeafNode* leaf = static_cast<LearnedLeafNode*>(node);
            new_leaf->model = leaf->model;
            new_leaf->slotdata = leaf->slotdata;
            new_leaf->buffer_slotdata = leaf->buffer_slotdata;
            new_leaf->parent_node = leaf->parent_node;
            new_leaf->prev_leaf = leaf->prev_leaf;
            new_leaf->next_leaf = leaf->next_leaf;
            new_leaf->level = leaf->level;
        } else if (node->is_leafnode()) {
            new_node = new LeafNode();
            LeafNode* new_leaf = static_cast<LeafNode*>(new_node);
            LeafNode* leaf = static_cast<LeafNode*>(node);
            new_leaf->slotdata = leaf->slotdata;
            new_leaf->parent_node = leaf->parent_node;
            new_leaf->prev_leaf = leaf->prev_leaf;
            new_leaf->next_leaf = leaf->next_leaf;
            new_leaf->level = leaf->level;
        } else {
            new_node = new InnerNode(node->level);
            InnerNode* new_inner = static_cast<InnerNode*>(new_node);
            InnerNode* inner = static_cast<InnerNode*>(node);
            new_inner->slotkey = inner->slotkey;
            new_inner->child_nodes = inner->child_nodes;
            new_inner->parent_node = inner->parent_node;
        }
        return new_node;
    }

public:
    std::pair<std::vector<Node *>*, std::vector<Node *>*> get_retrain_path_snapshot(LearnedLeafNode* node) {
        auto path = new std::vector<Node *>();
        auto original_path = new std::vector<Node *>();
        path->reserve(root->level + 1);
        Node* curr = node;
        int max_push_up = (node->slotuse() + FANOUT - 1) / FANOUT;
        while (curr != nullptr) {
            curr->is_locked = true;
            Node* snapshot = copy_node(curr);
            path->push_back(snapshot);
            original_path->push_back(curr);
            curr = curr->parent_node;
            if (curr) {
                if (curr->remain() >= max_push_up) {
                    // In order to replace a sub-tree for further update
                    curr->is_locked = true;
                    Node* parent = copy_node(curr);
                    path->push_back(parent);
                    original_path->push_back(curr);
                    break;
                }
                max_push_up = (max_push_up - curr->remain() + FANOUT - 1) / FANOUT;
            }
        }

        return std::make_pair(path, original_path);
    }

    void update_path_node_ptr(std::vector<Node *>* path) {
        for (int i = 0; i < path->size() - 1; ++i) {
            path->at(i)->parent_node = path->at(i+1);
            // Update child node ptr
            unsigned int pos = find_lower(path->at(i+1), path->at(i)->key(0));
            static_cast<InnerNode*>(path->at(i+1))->child_nodes[pos] = path->at(i);
        }
    }

    void detect_and_retrain_model_leaf(LearnedLeafNode* node, bool verbose = false, std::ostream* os = &std::cout) {
        // if (node->buffer_size() > RETRAIN_THRESHOLD && !verbose) {
        //     retrain_model_node(node);
        // } else if (node->buffer_size() > RETRAIN_THRESHOLD && verbose) {
        //     auto height = this->root->level;
        //     auto buffer_size = node->buffer_size();
        //     auto start = std::chrono::high_resolution_clock::now();
        //     retrain_model_node(node);
        //     auto end = std::chrono::high_resolution_clock::now();
        //     std::chrono::duration<double, std::nano> duration = end - start;
        //
        //     std::string msg = fmt::format("buffer size {}, height {}, retrain {} ns", buffer_size, height, duration.count());
        //     *os << msg << std::endl;
        // }
        // clear_root_buffer();
    }

    void clear_root_buffer() {
        for (auto key : this->root_buffer) {
            this->insert(key);
        }
    }

    void retrain_model_node(LearnedLeafNode* node) {
        auto [path, original] = get_retrain_path_snapshot(node);
        update_path_node_ptr(path);

        LearnedLeafNode* curr = static_cast<LearnedLeafNode*>(path->front());
        curr->delete_model();
        curr->model = new LinearModel<key_type, unsigned int>(MAX_EPSILON);

        // Merge slodata and buffer data of model leaf
        uint32_t slotdata_num = curr->slotdata.size();

        std::vector<key_type> merged_data(curr->slotdata.size() + curr->buffer_slotdata.size());
        std::sort(curr->buffer_slotdata.begin(), curr->buffer_slotdata.end());
        std::merge(curr->slotdata.begin(), curr->slotdata.end(), curr->buffer_slotdata.begin(),
                   curr->buffer_slotdata.end(), merged_data.begin());
        curr->buffer_slotdata.clear();
        curr->slotdata = merged_data;

        // Check if new model leaf will overflow
        std::pair<LearnedLeafNode*, LearnedLeafNode*> new_leaf_pair = std::make_pair(curr, nullptr);
        if (curr->slotuse() > model_leaf_slotmax) {
            // Split thez Angus  model leaf
            size_t shiftnum = curr->slotuse() >> 1;
            new_leaf_pair.second = new LearnedLeafNode();
            std::copy(std::make_move_iterator(curr->slotdata.begin() + shiftnum),
                      std::make_move_iterator(curr->slotdata.end()), std::back_inserter(new_leaf_pair.second->slotdata));
            new_leaf_pair.first->slotdata.resize(shiftnum);
            insert_node_to_subtree(new_leaf_pair.second, static_cast<InnerNode*>(curr->parent_node), true,
                                   new_leaf_pair.first->key(new_leaf_pair.first->slotuse() - 1));
        }

        // Try to retrain the model of first node
        retrain_and_split_model_leaf(new_leaf_pair.first);

        // Try to retrain the model of second node
        if (new_leaf_pair.second) {
            retrain_and_split_model_leaf(new_leaf_pair.second);
        }

        // Replace b+-tree node with new subtree root
        replace_new_node_to_tree(path, original);

        delete path;
        delete original;
    }

    void replace_new_node_to_tree(std::vector<Node *>* path, std::vector<Node *>* original_path) {
        for (int i = 0; i < path->size() - 1; ++i) {
            // Update child node ptr
            InnerNode* parent = static_cast<InnerNode*>(path->at(i + 1));
            for (int q = 0; q <= parent->slotuse(); ++q) {
                parent->child_nodes[q]->parent_node = parent;
            }
        }
        if (path->size() != original_path->size()) {
            root = path->back();
        } else {
            // Replace the subtree root node
            if (original_path->back() == root) {
                root = path->back();
            } else {
                InnerNode* parent = static_cast<InnerNode*>(path->back());
                InnerNode* original_parent = static_cast<InnerNode*>(original_path->back());
                unsigned int pos = find_lower(original_parent->parent_node, original_parent->key(0));
                static_cast<InnerNode*>(original_parent->parent_node)->child_nodes[pos] = parent;
            }

            if (original_path->front() == head_leaf) {
                head_leaf = static_cast<LeafNode*>(path->front());
            } else if (original_path->front() == tail_leaf) {
                tail_leaf = static_cast<LeafNode*>(path->front())->next_leaf? static_cast<LeafNode*>(path->front())->next_leaf : static_cast<LeafNode*>(path->front());
            }
        }

        // Delete original path
        for (int i = 0; i < original_path->size(); ++i) {
            free_node(original_path->at(i));
        }
    }

    void retrain_and_split_model_leaf(LearnedLeafNode* model_leaf) {
        LearnedLeafNode* curr = model_leaf;
        while (true) {
            int i = 0;
            bool need_get_model = true;
            for (i = 0; i < curr->slotuse(); ++i) {
                bool is_added = curr->model->add_point(curr->key(i), i);
                if (!is_added) {
                    int left = curr->slotuse() - i;
                    if (i < model_leaf_slotmin && left <= leaf_slotmin) {
                        // Last leaf that can't be added to the model
                        // Rest data cannot even form a normal leaf
                        i = curr->slotuse();
                        curr->delete_model();
                        need_get_model = false;
                        break;
                    } else if (i < model_leaf_slotmin){
                        i = leaf_slotmin;
                        curr->delete_model();
                    } else {
                        if (left <= leaf_slotmin) {
                            // Last leaf that can't be formed
                            if (i - leaf_slotmin < model_leaf_slotmin) {
                                i = curr->slotuse();
                                curr->delete_model();
                                need_get_model = false;
                                break;
                            }
                            i -= leaf_slotmin;
                        } else {
                            curr->get_model_param();
                        }
                    }
                    LearnedLeafNode* new_leaf = new LearnedLeafNode();
                    new_leaf->slotdata.resize(curr->slotuse() - i);
                    std::copy(std::make_move_iterator(curr->slotdata.begin() + i),
                              std::make_move_iterator(curr->slotdata.end()), std::back_inserter(new_leaf->slotdata));
                    curr->slotdata.resize(i);
                    insert_node_to_subtree(new_leaf, static_cast<InnerNode*>(curr->parent_node), true,
                                           curr->key(curr->slotuse() - 1));
                    curr = new_leaf;
                    break;
                }
            }
            if (i == curr->slotuse()) {
                if (need_get_model) {
                    curr->get_model_param();
                }
                break;
            }
        }
    }

    void insert_node_to_subtree(Node* new_node, InnerNode* parent, bool has_split_key, key_type pre_split_key) {
        if (parent->is_full()) {
            int insert_pos = find_lower(parent, new_node->key(new_node->slotuse() - 1));
            bool is_left = insert_pos < FANOUT / 2;

            key_type split_key;
            auto [left, right] = split_inner_node(parent, split_key);

            if (left->parent_node) {
                insert_node_to_subtree(right, static_cast<InnerNode*>(left->parent_node), true, split_key);
            } else {
                InnerNode* new_root = new InnerNode(left->level + 1);
                new_root->slotkey.emplace_back(split_key);
                new_root->child_nodes.emplace_back(left);
                new_root->child_nodes.emplace_back(right);
                left->parent_node = new_root;
                right->parent_node = new_root;
                return;
            }

            // Insert the new node to the parent node (left or right)
            if (is_left) {
                insert_node_to_subtree(new_node, left, has_split_key, pre_split_key);
            } else {
                insert_node_to_subtree(new_node, right, has_split_key, pre_split_key);
            }
        } else {
            int insert_pos;
            if (has_split_key) {
                insert_pos = find_lower(parent, pre_split_key);
                parent->slotkey.insert(parent->slotkey.begin() + insert_pos, pre_split_key);
                parent->child_nodes.insert(parent->child_nodes.begin() + insert_pos + 1, new_node);
                if (new_node->is_leafnode()) {
                    update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos]),
                                        static_cast<LeafNode*>(new_node),
                                        static_cast<LeafNode*>(parent->child_nodes[insert_pos])->next_leaf);
                }
            } else {
                insert_pos = find_lower(parent, new_node->key(new_node->slotuse() - 1));
                if (insert_pos == parent->slotuse() && new_node->key(0) > parent->child_nodes[insert_pos]->key(0)) {
                    // new node is the largest one
                    parent->slotkey.emplace_back(
                            parent->child_nodes[insert_pos]->key(parent->child_nodes[insert_pos]->slotuse() - 1));
                    parent->child_nodes.emplace_back(new_node);

                    update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos]),
                                        static_cast<LeafNode*>(new_node),
                                        static_cast<LeafNode*>(parent->child_nodes[insert_pos])->next_leaf);
                } else if (insert_pos == 0) {



         unsigned int key_index = parent->child_nodes[insert_pos]->slotuse() > 0
                                                     ? parent->child_nodes[insert_pos]->slotuse() - 1
                                                     : 0;
                    if (parent->child_nodes[insert_pos]->key(key_index) < new_node->key(0)) {
                        parent->slotkey.insert(parent->slotkey.begin() + insert_pos,
                                               parent->child_nodes[insert_pos]->key(key_index));
                        parent->child_nodes.insert(parent->child_nodes.begin() + insert_pos + 1, new_node);

                        update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos]),
                                            static_cast<LeafNode*>(new_node),
                                            static_cast<LeafNode*>(parent->child_nodes[insert_pos])->next_leaf);
                    } else {
                        parent->slotkey.insert(parent->slotkey.begin() + insert_pos,
                                               new_node->key(new_node->slotuse() - 1));
                        parent->child_nodes.insert(parent->child_nodes.begin() + insert_pos, new_node);

                        update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos + 1])->prev_leaf,
                                            static_cast<LeafNode*>(new_node),
                                            static_cast<LeafNode*>(parent->child_nodes[insert_pos + 1]));
                    }
                } else {
                    parent->slotkey.insert(parent->slotkey.begin() + insert_pos,
                                           new_node->key(new_node->slotuse() - 1));
                    parent->child_nodes.insert(parent->child_nodes.begin() + insert_pos, new_node);

                    update_leaves_chain(static_cast<LeafNode*>(parent->child_nodes[insert_pos + 1])->prev_leaf,
                                        static_cast<LeafNode*>(new_node),
                                        static_cast<LeafNode*>(parent->child_nodes[insert_pos + 1]));
                }
            }
            new_node->parent_node = parent;
        }
    }
};

#endif  // VECBPLUSTREE_HPP
