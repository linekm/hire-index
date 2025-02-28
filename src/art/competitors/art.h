#pragma once

#include <assert.h>
#include <emmintrin.h>  // x86 SSE intrinsics
#include <stdint.h>     // integer types
#include <stdio.h>
#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <sys/time.h>  // gettime
#include <type_traits>

#include <algorithm>  // std::random_shuffle
#include <utility>
#include <vector>
#include <string>
#include <stack>

#include "../util.h"
#include "base.h"

// Uses ART as a non-clustered primary index that stores <key, offset> pairs.
// At lookup time, we retrieve a key's offset from ART and lookup its value in
// the extra database.
namespace tli_art{

template<class KeyType>
class ART : public Base<KeyType> {
 public:
  ART(){}
  ~ART() {
    if (tree_) {
      destructTree(tree_);
    }
  }

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
    std::vector<KeyValue<std::string>> transform_data(data.size());

    return util::timing([&] {
      for (size_t i = 0; i < data.size(); ++i) {
        util::convert2String(data[i].key, transform_data[i].key);
        Element<std::string>* e = new Element<std::string>(transform_data[i].key, data[i].value);
        transform_data[i].value = (uint64_t(e) >> 1);
      }
      bulk_insert(&tree_, transform_data, 0, data.size(), 0);
    });
  }

  size_t EqualityLookup(const KeyType& lookup_key, uint32_t thread_id) const {
    std::string key;
    util::convert2String(lookup_key, key);

    Node* node = lookup(tree_, static_cast<const uint8_t*>(static_cast<const void*>(key.c_str())), key.length(), 0);
    if (node){
      return reinterpret_cast<Element<std::string>*>(getLeafValue(node) << 1)->value;
    }
    return util::NOT_FOUND;
  }

  uint64_t RangeQuery(const KeyType& lower_key, const KeyType& upper_key) const {
    std::string lkey, ukey;
    util::convert2String(lower_key, lkey);
    util::convert2String(upper_key, ukey);

    Iterator it;
    uint64_t result = 0;
    if (bound(tree_, static_cast<const uint8_t*>(static_cast<const void*>(lkey.c_str())), lkey.length(), it, true)){
      do{
        result += reinterpret_cast<Element<std::string>*>(it.value << 1)->value;
      } while(iteratorNext(it) && !(ukey < loadKey(it.value)));
    }
    return result;
  }

  void Erase(const KeyType& key) {
    std::string key_str;
    util::convert2String(key, key_str);
    erase(tree_, &tree_, static_cast<const uint8_t*>(static_cast<const void*>(key_str.c_str())), key_str.size(), 0);
  }

  void Insert(const KeyValue<KeyType>& data) {
    std::string key;
    util::convert2String(data.key, key);
    Element<std::string>* e = new Element<std::string>(key, data.value);

    insert(tree_, &tree_, static_cast<const uint8_t*>(static_cast<const void*>(key.c_str())), key.length(), 0, uint64_t(e) >> 1);
  }

  std::string name() const { return "ART"; }

  std::size_t size() const { return size_in_bytes(tree_); }

  bool applicable(bool unique, bool range_query, bool insert, bool multithread, const std::string& _ops_filename) const {
    return unique && !multithread;
  }

 private:

  const std::string& loadKey(uintptr_t tid) const {
    return reinterpret_cast<Element<std::string>*>(tid << 1)->key;
  }

  // Constants for the node types
  static const int8_t NodeType4 = 0;
  static const int8_t NodeType16 = 1;
  static const int8_t NodeType48 = 2;
  static const int8_t NodeType256 = 3;

  // The maximum prefix length for compressed paths stored in the
  // header, if the path is longer it is loaded from the database on
  // demand
  static const unsigned maxPrefixLength = 8;

  // Shared header of all inner nodes
  struct Node {
    // length of the compressed path (prefix)
    uint32_t prefixLength;
    // number of non-null children
    uint16_t count;
    // node type
    int8_t type;
    // compressed path (prefix)
    uint8_t prefix[maxPrefixLength];

    Node(int8_t type) : prefixLength(0), count(0), type(type) {}
  };

  // Node with up to 4 children
  struct Node4 : Node {
    uint8_t key[4];
    Node* child[4];

    Node4() : Node(NodeType4) {
      memset(key, 0, sizeof(key));
      memset(child, 0, sizeof(child));
    }
  };

  // Node with up to 16 children
  struct Node16 : Node {
    uint8_t key[16];
    Node* child[16];

    Node16() : Node(NodeType16) {
      memset(key, 0, sizeof(key));
      memset(child, 0, sizeof(child));
    }
  };

  static const uint8_t emptyMarker = 48;

  // Node with up to 48 children
  struct Node48 : Node {
    uint8_t childIndex[256];
    Node* child[48];

    Node48() : Node(NodeType48) {
      memset(childIndex, emptyMarker, sizeof(childIndex));
      memset(child, 0, sizeof(child));
    }
  };

  // Node with up to 256 children
  struct Node256 : Node {
    Node* child[256];

    Node256() : Node(NodeType256) {
      memset(child, 0, sizeof(child));
    }
  };

  __m128i static forceinline _mm_cmpge_epu8(__m128i a, __m128i b) {
    return _mm_cmpeq_epi8(_mm_max_epu8(a, b), a);
  }

  inline Node* makeLeaf(uintptr_t tid) {
    // Create a pseudo-leaf
    return reinterpret_cast<Node*>((tid << 1) | 1);
  }

  inline bool isLeaf(Node* node) const {
    // Is the node a leaf?
    return reinterpret_cast<uintptr_t>(node) & 1;
  }

  inline uintptr_t getLeafValue(Node* node) const {
    // The the value stored in the pseudo-leaf
    return reinterpret_cast<uintptr_t>(node) >> 1;
  }

  struct IteratorEntry {
    Node* node;
    int pos;
  };

  struct Iterator {
    /// The current value, valid if stack is not empty
    uint64_t value;
    /// Stack, actually the size is determined at runtime
    std::stack<IteratorEntry> stack;
  };

  static inline unsigned ctz(uint16_t x) {
    // Count trailing zeros, only defined for x>0
#ifdef __GNUC__
    return __builtin_ctz(x);
#else
    // Adapted from Hacker's Delight
    unsigned n = 1;
    if ((x & 0xFF) == 0) {
      n += 8;
      x = x >> 8;
    }
    if ((x & 0x0F) == 0) {
      n += 4;
      x = x >> 4;
    }
    if ((x & 0x03) == 0) {
      n += 2;
      x = x >> 2;
    }
    return n - (x & 1);
#endif
  }

  Node** findChild(Node* n, uint8_t keyByte) const {
    // This address is used to communicate that search failed
    static Node* nullNode = NULL;
    // Find the next child for the keyByte
    switch (n->type) {
      case NodeType4: {
        Node4* node = static_cast<Node4*>(n);
        for (unsigned i = 0; i < node->count; i++)
          if (node->key[i] == keyByte) return &node->child[i];
        return &nullNode;
      }
      case NodeType16: {
        Node16* node = static_cast<Node16*>(n);
        __m128i cmp = _mm_cmpeq_epi8(
            _mm_set1_epi8(keyByte),
            _mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
        unsigned bitfield = _mm_movemask_epi8(cmp) & ((1 << node->count) - 1);
        if (bitfield)
          return &node->child[ctz(bitfield)];
        else
          return &nullNode;
      }
      case NodeType48: {
        Node48* node = static_cast<Node48*>(n);
        if (node->childIndex[keyByte] != emptyMarker)
          return &node->child[node->childIndex[keyByte]];
        else
          return &nullNode;
      }
      case NodeType256: {
        Node256* node = static_cast<Node256*>(n);
        return &(node->child[keyByte]);
      }
    }
    throw;  // Unreachable
  }

  bool iteratorNext(Iterator& iter) const {
    // Skip leaf
    if (!iter.stack.empty() && (isLeaf(iter.stack.top().node))) iter.stack.pop();

    // Look for next leaf
    while (!iter.stack.empty()) {
      auto entry = iter.stack.top();
      Node* node = entry.node;

      // Leaf found
      if (isLeaf(node)) {
        iter.value = getLeafValue(node);
        return true;
      }

      // Find next node
      iter.stack.pop();
      Node* next = nullptr;
      switch (node->type) {
        case NodeType4: {
          Node4* n = static_cast<Node4*>(node);
          if (entry.pos < node->count)
            next = n->child[entry.pos++];
          break;
        }
        case NodeType16: {
          Node16* n = static_cast<Node16*>(node);
          if (entry.pos < node->count)
            next = n->child[entry.pos++];
          break;
        }
        case NodeType48: {
          Node48* n = static_cast<Node48*>(node);
          for (; entry.pos < 256; entry.pos++)
            if (n->childIndex[entry.pos] != emptyMarker) {
              next = n->child[n->childIndex[entry.pos++]];
              break;
            }
          break;
        }
        case NodeType256: {
          Node256* n = static_cast<Node256*>(node);
          for (; entry.pos < 256; entry.pos++)
            if (n->child[entry.pos]) {
              next = n->child[entry.pos++];
              break;
            }
          break;
        }
      }

      if (next) {
        iter.stack.push(entry);
        iter.stack.push({next, 0});
      }
    }

    return false;
  }

  Node* minimum(Node* node) const {
    // Find the leaf with smallest key
    if (!node) return NULL;

    if (isLeaf(node)) return node;

    switch (node->type) {
      case NodeType4: {
        Node4* n = static_cast<Node4*>(node);
        return minimum(n->child[0]);
      }
      case NodeType16: {
        Node16* n = static_cast<Node16*>(node);
        return minimum(n->child[0]);
      }
      case NodeType48: {
        Node48* n = static_cast<Node48*>(node);
        unsigned pos = 0;
        while (n->childIndex[pos] == emptyMarker) pos++;
        return minimum(n->child[n->childIndex[pos]]);
      }
      case NodeType256: {
        Node256* n = static_cast<Node256*>(node);
        unsigned pos = 0;
        while (!n->child[pos]) pos++;
        return minimum(n->child[pos]);
      }
    }
    throw;  // Unreachable
  }

  Node* maximum(Node* node) {
    // Find the leaf with largest key
    if (!node) return NULL;

    if (isLeaf(node)) return node;

    switch (node->type) {
      case NodeType4: {
        Node4* n = static_cast<Node4*>(node);
        return maximum(n->child[n->count - 1]);
      }
      case NodeType16: {
        Node16* n = static_cast<Node16*>(node);
        return maximum(n->child[n->count - 1]);
      }
      case NodeType48: {
        Node48* n = static_cast<Node48*>(node);
        unsigned pos = 255;
        while (n->childIndex[pos] == emptyMarker) pos--;
        return maximum(n->child[n->childIndex[pos]]);
      }
      case NodeType256: {
        Node256* n = static_cast<Node256*>(node);
        unsigned pos = 255;
        while (!n->child[pos]) pos--;
        return maximum(n->child[pos]);
      }
    }
    throw;  // Unreachable
  }

  bool leafMatches(Node* leaf, const uint8_t key[], unsigned keyLength,
                   unsigned depth) const {
    // Check if the key of the leaf is equal to the searched key
    const std::string& leafKey = loadKey(getLeafValue(leaf));
    if (leafKey.length() != keyLength || memcmp(leafKey.c_str() + depth, key + depth, keyLength - depth)){
      return false;
    }
    return true;
  }

  bool leafPrefixMatches(Node* leaf, const uint8_t key[], unsigned keyLength,
                   unsigned depth) const {
    // Check if the key of the leaf is equal to the searched key
    const std::string& leafKey = loadKey(getLeafValue(leaf));
    if (leafKey.length() < keyLength || memcmp(leafKey.c_str() + depth, key + depth, keyLength - depth)){
      return false;
    }
    return true;
  }

  unsigned prefixMismatch(const uint8_t key1[], unsigned l1,
                          const uint8_t key2[], unsigned l2,
                          unsigned depth) const {
    unsigned compBytes = std::min(l1, l2) - depth;
    unsigned pos;
    for (pos=0; pos < compBytes; pos++) {
      if (key1[depth + pos] != key2[depth + pos])
        return pos;
    }
    return pos;
  }

  unsigned prefixMismatch(Node* node, const uint8_t key[], unsigned keyLength,
                          unsigned depth) const
  // Compare the the key with the prefix, return the number matching bytes
  {
    unsigned compBytes = std::min(keyLength - depth, node->prefixLength);
    unsigned pos = 0;
    if (compBytes > maxPrefixLength) {
      for (; pos < maxPrefixLength; pos++)
        if (key[depth + pos] != node->prefix[pos]) return pos;

      // Load key from database
      Node* minNode = minimum(node);
      const std::string& minKey = loadKey(getLeafValue(minNode));
      for (; pos < compBytes; pos++)
        if (key[depth + pos] != (uint8_t)minKey[depth + pos]) return pos;
    } else {
      for (; pos < compBytes; pos++)
        if (key[depth + pos] != node->prefix[pos]) return pos;
    }
    return pos;
  }

  bool bound(Node* n, const uint8_t key[], unsigned keyLength, Iterator& iterator,
             bool lower = true) const {
    iterator.stack = std::stack<IteratorEntry>();
    if (!n) return false;

    unsigned depth = 0;
    while (true) {
      IteratorEntry entry;
      entry.node = n;

      if (isLeaf(n)) {
        iterator.stack.push(entry);
        iterator.value = getLeafValue(n);

        const std::string& leafKey = loadKey(getLeafValue(n));
        unsigned i;
        for (i = depth; i < min(keyLength, leafKey.length()); i++){
          if ((uint8_t)leafKey[i] != key[i]) {
            if ((uint8_t)leafKey[i] < key[i]) {
              // Less
              return iteratorNext(iterator);
            }
            // Greater
            return true;
          }
        }
        // Equal
        // Ensure key must not be prefix of other keys.
        if (lower)
          return true;
        else
          return iteratorNext(iterator);
      }

      if (n->prefixLength) {
        unsigned mismatchPos = prefixMismatch(n, key, keyLength, depth);
        if (mismatchPos != n->prefixLength) {
          uint8_t keyByte;
          if (mismatchPos < maxPrefixLength){
            keyByte = n->prefix[mismatchPos];
          }
          else{
            const std::string& minKey = loadKey(getLeafValue(minimum(n)));
            keyByte = minKey[depth + mismatchPos];
          }
          if (keyByte < key[depth + mismatchPos]) {
            // Less
            return iteratorNext(iterator);
          }
          // Greater
          entry.pos = 0;
          iterator.stack.push(entry);
          return iteratorNext(iterator);
        }
        depth += n->prefixLength;
      }
      uint8_t keyByte = key[depth];

      Node* next = nullptr;
      switch (n->type) {
        case NodeType4: {
          Node4* node = static_cast<Node4*>(n);
          for (entry.pos = 0; entry.pos < node->count; entry.pos++){
            if (node->key[entry.pos] == keyByte) {
              next = node->child[entry.pos++];
              break;
            } else if (node->key[entry.pos] > keyByte)
              break;
          }
          break;
        }
        case NodeType16: {
          Node16* node = static_cast<Node16*>(n);
          __m128i cmp = _mm_cmpge_epu8(
              _mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)), _mm_set1_epi8(keyByte));
          unsigned bitfield = (_mm_movemask_epi8(cmp) & ((1 << node->count) - 1));
          if (bitfield){
            entry.pos = ctz(bitfield);
            if (node->key[entry.pos] == keyByte){
              next = node->child[entry.pos++];
            }
          }
          else{
            entry.pos = node->count;
          }
          break;
        }
        case NodeType48: {
          Node48* node = static_cast<Node48*>(n);
          entry.pos = keyByte;
          if (node->childIndex[entry.pos++] != emptyMarker) {
            next = node->child[node->childIndex[keyByte]];
            break;
          }
          break;
        }
        case NodeType256: {
          Node256* node = static_cast<Node256*>(n);
          entry.pos = keyByte;
          next = node->child[entry.pos++];
          break;
        }
      }

      if (!next){
        iterator.stack.push(entry);
        return iteratorNext(iterator);
      }

      iterator.stack.push(entry);
      n = next;
      ++depth;
    }
  }

  Node* lookup(Node* node, const uint8_t key[], unsigned keyLength, unsigned depth) const {
    // Find the node with a matching key, optimistic version

    bool skippedPrefix =
        false;  // Did we optimistically skip some prefix without checking it?

    while (node) {
      if (isLeaf(node)) {
        // Check leaf
        if (skippedPrefix){
          if (!leafMatches(node, key, keyLength, 0)){
            return NULL;
          }
        }
        else{
          if (!leafMatches(node, key, keyLength, depth)){
            return NULL;
          }
        }
        return node;
      }

      if (node->prefixLength) {
        if (node->prefixLength <= maxPrefixLength) {
          if (prefixMismatch(node, key, keyLength, depth) != node->prefixLength){
            return NULL;
          }
        } else
          skippedPrefix = true;
        depth += node->prefixLength;
      }

      node = *findChild(node, key[depth]);
      depth++;
    }

    return NULL;
  }

  Node* lookupPessimistic(Node* node, const uint8_t key[], unsigned keyLength,
                          unsigned depth) {
    // Find the node with a matching key, alternative pessimistic version

    while (node != NULL) {
      if (isLeaf(node)) {
        if (leafMatches(node, key, keyLength, depth)) return node;
        return NULL;
      }

      if (node->prefixLength){
        if (prefixMismatch(node, key, keyLength, depth) != node->prefixLength)
          return NULL;
        depth += node->prefixLength;
      }

      node = *findChild(node, key[depth]);
      depth++;
    }

    return NULL;
  }

  Node* lookupPrefix(Node* node, const uint8_t* key, uint32_t keyLength,
                     unsigned depth) const {
    while(node){
      if (depth == keyLength){
        return node;
      }

      if (isLeaf(node)) {
        // Check if it matches
        if (!leafPrefixMatches(node, key, keyLength, depth)){
          return nullptr;
        }
        return node;
      }

      if (node->prefixLength){
        unsigned misMatchPos = prefixMismatch(node, key, keyLength, depth);
        if (depth + misMatchPos == keyLength){
          return node;
        }
        if (misMatchPos != node->prefixLength){
          return nullptr;
        }
        depth += node->prefixLength;
      }
      node = *findChild(node, key[depth]);
      ++depth;
    }
    return nullptr;
  }

  bool lookupPrefixMin(Node* root, const uint8_t* key, uint32_t keyLength,
                       Iterator& iterator) {
    Node* node = lookupPrefix(root, key, keyLength, 0);

    if (node) {
      iterator.stack = std::stack<IteratorEntry>();
      iterator.stack.push({node, 0});
      // Descend to first leaf if necessary
      if (isLeaf(node))
        iterator.value = getLeafValue(node);
      else
        return iteratorNext(iterator);  // xxx
      return true;
    } else
      return false;
  }

  // Forward references
  // void insertNode4(Node4* node, Node** nodeRef, uint8_t keyByte, Node*
  // child); void insertNode16(Node16* node, Node** nodeRef, uint8_t keyByte,
  // Node* child); void insertNode48(Node48* node, Node** nodeRef, uint8_t
  // keyByte, Node* child); void insertNode256(Node256* node, Node** nodeRef,
  // uint8_t keyByte, Node* child);

  unsigned min(unsigned a, unsigned b) const {
    // Helper function
    return (a < b) ? a : b;
  }

  void copyPrefix(Node* src, Node* dst) {
    // Helper function that copies the prefix from the source to the destination
    // node
    dst->prefixLength = src->prefixLength;
    memcpy(dst->prefix, src->prefix, min(src->prefixLength, maxPrefixLength));
  }

  void insert(Node* node, Node** nodeRef, const uint8_t key[], unsigned keyLength, unsigned depth,
              uintptr_t value) {
    // Insert the leaf value into the tree

    if (node == NULL) {
      *nodeRef = makeLeaf(value);
      return;
    }

    if (isLeaf(node)) {
      // Replace leaf with Node4 and store both leaves in it
      const std::string& existingKey = loadKey(getLeafValue(node));
      unsigned newPrefixLength = prefixMismatch(static_cast<const uint8_t*>(static_cast<const void*>(existingKey.c_str())), existingKey.length(), key, keyLength, depth);

      Node4* newNode = new Node4();
      newNode->prefixLength = newPrefixLength;
      memcpy(newNode->prefix, key + depth,
             min(newPrefixLength, maxPrefixLength));
      *nodeRef = newNode;

      insertNode4(newNode, nodeRef, existingKey[depth + newPrefixLength], node);
      insertNode4(newNode, nodeRef, key[depth + newPrefixLength],
                  makeLeaf(value));
      return;
    }

    // Handle prefix of inner node
    if (node->prefixLength) {
      unsigned mismatchPos = prefixMismatch(node, key, keyLength, depth);
      if (mismatchPos != node->prefixLength) {
        // Prefix differs, create new node
        Node4* newNode = new Node4();
        *nodeRef = newNode;
        newNode->prefixLength = mismatchPos;
        memcpy(newNode->prefix, node->prefix,
               min(mismatchPos, maxPrefixLength));
        // Break up prefix
        if (node->prefixLength < maxPrefixLength) {
          insertNode4(newNode, nodeRef, node->prefix[mismatchPos], node);
          node->prefixLength -= (mismatchPos + 1);
          memmove(node->prefix, node->prefix + mismatchPos + 1,
                  min(node->prefixLength, maxPrefixLength));
        } else {
          node->prefixLength -= (mismatchPos + 1);
          const std::string& minKey = loadKey(getLeafValue(minimum(node)));
          insertNode4(newNode, nodeRef, minKey[depth + mismatchPos], node);
          memcpy(node->prefix, minKey.c_str() + depth + mismatchPos + 1,
                  min(node->prefixLength, maxPrefixLength));
        }
        insertNode4(newNode, nodeRef, key[depth + mismatchPos],
                    makeLeaf(value));
        return;
      }
      depth += node->prefixLength;
    }

    // Recurse
    Node** child = findChild(node, key[depth]);
    if (*child) {
      insert(*child, child, key, keyLength, depth + 1, value);
      return;
    }

    // Insert leaf into inner node
    Node* newNode = makeLeaf(value);
    switch (node->type) {
      case NodeType4:
        insertNode4(static_cast<Node4*>(node), nodeRef, key[depth], newNode);
        break;
      case NodeType16:
        insertNode16(static_cast<Node16*>(node), nodeRef, key[depth], newNode);
        break;
      case NodeType48:
        insertNode48(static_cast<Node48*>(node), nodeRef, key[depth], newNode);
        break;
      case NodeType256:
        insertNode256(static_cast<Node256*>(node), nodeRef, key[depth],
                      newNode);
        break;
    }
  }

  void insertNode4(Node4* node, Node** nodeRef, uint8_t keyByte, Node* child) {
    // Insert leaf into inner node
    if (node->count < 4) {
      // Insert element
      unsigned pos;
      for (pos = 0; (pos < node->count) && (node->key[pos] < keyByte); pos++)
        ;
      memmove(node->key + pos + 1, node->key + pos, node->count - pos);
      memmove(node->child + pos + 1, node->child + pos,
              (node->count - pos) * sizeof(uintptr_t));
      node->key[pos] = keyByte;
      node->child[pos] = child;
      node->count++;
    } else {
      // Grow to Node16
      Node16* newNode = new Node16();
      *nodeRef = newNode;
      newNode->count = 4;
      copyPrefix(node, newNode);
      memcpy(newNode->key, node->key, node->count * sizeof(uint8_t));
      memcpy(newNode->child, node->child, node->count * sizeof(uintptr_t));
      delete node;
      return insertNode16(newNode, nodeRef, keyByte, child);
    }
  }

  void insertNode16(Node16* node, Node** nodeRef, uint8_t keyByte,
                    Node* child) {
    // Insert leaf into inner node
    if (node->count < 16) {
      __m128i cmp = _mm_cmpge_epu8(
          _mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)), _mm_set1_epi8(keyByte));
      unsigned bitfield = (_mm_movemask_epi8(cmp) & ((1 << node->count) - 1));
      // Insert element
      unsigned pos;
      if (bitfield){
        pos = ctz(bitfield);
      }
      else{
        pos = node->count;
      }
      memmove(node->key + pos + 1, node->key + pos, node->count - pos);
      memmove(node->child + pos + 1, node->child + pos,
              (node->count - pos) * sizeof(uintptr_t));
      node->key[pos] = keyByte;
      node->child[pos] = child;
      node->count++;
    } else {
      // Grow to Node48
      Node48* newNode = new Node48();
      *nodeRef = newNode;
      memcpy(newNode->child, node->child, node->count * sizeof(uintptr_t));
      for (unsigned i = 0; i < node->count; i++)
        newNode->childIndex[node->key[i]] = i;
      copyPrefix(node, newNode);
      newNode->count = node->count;
      delete node;
      return insertNode48(newNode, nodeRef, keyByte, child);
    }
  }

  void insertNode48(Node48* node, Node** nodeRef, uint8_t keyByte,
                    Node* child) {
    // Insert leaf into inner node
    if (node->count < 48) {
      // Insert element
      unsigned pos = node->count;
      if (node->child[pos])
        for (pos = 0; node->child[pos] != NULL; pos++)
          ;
      node->child[pos] = child;
      node->childIndex[keyByte] = pos;
      node->count++;
    } else {
      // Grow to Node256
      Node256* newNode = new Node256();
      for (unsigned i = 0; i < 256; i++)
        if (node->childIndex[i] != 48)
          newNode->child[i] = node->child[node->childIndex[i]];
      newNode->count = node->count;
      copyPrefix(node, newNode);
      *nodeRef = newNode;
      delete node;
      return insertNode256(newNode, nodeRef, keyByte, child);
    }
  }

  void insertNode256(Node256* node, Node** nodeRef, uint8_t keyByte,
                     Node* child) {
    // Insert leaf into inner node
    node->count++;
    node->child[keyByte] = child;
  }

  void bulk_insert(Node** nodeRef, const std::vector<KeyValue<std::string>>& bulkVec, size_t first, size_t end, unsigned depth) {
    // Bulk insert leaf values into the tree

    // Empty array
    if (first == end) {
      *nodeRef = nullptr;
      return;
    }
    // Allocate new leaf
    if (end - first == 1){
      *nodeRef = makeLeaf(bulkVec[first].value);
      return;
    }

    unsigned curDepth = depth;
    while(true){
      // Partition by the byte curDepth
      std::vector<size_t> vec = {first};
      size_t cur = first;
      while(cur != end){
        size_t found = std::upper_bound(bulkVec.begin() + cur + 1, bulkVec.begin() + end, (uint8_t)(bulkVec[cur].key[curDepth]), [&](uint8_t c, const KeyValue<std::string>& e) {
                            return c < (uint8_t)(e.key[curDepth]);
                          }) - bulkVec.begin();
        vec.push_back(found);
        cur = found;
      }

      size_t count = vec.size() - 1;
      // Longer common prefix
      if (count == 1){
        ++curDepth;
        continue;
      }
      // Determine type of new node
      Node* newNode;
      if (count <= 4){
        newNode = new Node4();
        for (size_t i = 0; i < count; i ++){
          ((Node4*)newNode)->key[i] = bulkVec[vec[i]].key[curDepth];
          bulk_insert(&((Node4*)newNode)->child[i], bulkVec, vec[i], vec[i + 1], curDepth + 1);
        }
      }
      else if (count <= 16){
        newNode = new Node16();
        for (size_t i = 0; i < count; i ++){
          ((Node16*)newNode)->key[i] = bulkVec[vec[i]].key[curDepth];
          bulk_insert(&((Node16*)newNode)->child[i], bulkVec, vec[i], vec[i + 1], curDepth + 1);
        }
      }
      else if (count <= 48){
        newNode = new Node48();
        for (size_t i = 0; i < count; i ++){
          ((Node48*)newNode)->childIndex[(uint8_t)(bulkVec[vec[i]].key[curDepth])] = i;
          bulk_insert(&((Node48*)newNode)->child[i], bulkVec, vec[i], vec[i + 1], curDepth + 1);
        }
      }
      else {
        newNode = new Node256();
        for (unsigned i = 0; i < count; i++){
          bulk_insert(&((Node256*)newNode)->child[(uint8_t)(bulkVec[vec[i]].key[curDepth])], bulkVec, vec[i], vec[i + 1], curDepth + 1);
        }
      }
      *nodeRef = newNode;
      newNode->count = count;
      newNode->prefixLength = curDepth - depth;
      memcpy(newNode->prefix, bulkVec[first].key.c_str() + depth, min(curDepth - depth, maxPrefixLength));
      break;
    }
  }

  // Forward references
  // void eraseNode4(Node4* node, Node** nodeRef, Node** leafPlace);
  // void eraseNode16(Node16* node, Node** nodeRef, Node** leafPlace);
  // void eraseNode48(Node48* node, Node** nodeRef, uint8_t keyByte);
  // void eraseNode256(Node256* node, Node** nodeRef, uint8_t keyByte);

  void erase(Node* node, Node** nodeRef, const uint8_t key[], unsigned keyLength,
             unsigned depth) {
    // Delete a leaf from a tree

    if (!node) return;

    if (isLeaf(node)) {
      // Make sure we have the right leaf
      if (leafMatches(node, key, keyLength, depth))
        *nodeRef = NULL;
      return;
    }

    // Handle prefix
    if (node->prefixLength) {
      if (prefixMismatch(node, key, keyLength, depth) != node->prefixLength)
        return;
      depth += node->prefixLength;
    }

    Node** child = findChild(node, key[depth]);
    if (isLeaf(*child) &&
        leafMatches(*child, key, keyLength, depth)) {
      // Leaf found, delete it in inner node
      switch (node->type) {
        case NodeType4:
          eraseNode4(static_cast<Node4*>(node), nodeRef, child);
          break;
        case NodeType16:
          eraseNode16(static_cast<Node16*>(node), nodeRef, child);
          break;
        case NodeType48:
          eraseNode48(static_cast<Node48*>(node), nodeRef, key[depth]);
          break;
        case NodeType256:
          eraseNode256(static_cast<Node256*>(node), nodeRef, key[depth]);
          break;
      }
    } else {
      // Recurse
      erase(*child, child, key, keyLength, depth + 1);
    }
  }

  void eraseNode4(Node4* node, Node** nodeRef, Node** leafPlace) {
    // Delete leaf from inner node
    unsigned pos = leafPlace - node->child;
    memmove(node->key + pos, node->key + pos + 1, node->count - pos - 1);
    memmove(node->child + pos, node->child + pos + 1,
            (node->count - pos - 1) * sizeof(uintptr_t));
    node->count--;

    if (node->count == 1) {
      // Get rid of one-way node
      Node* child = node->child[0];
      if (!isLeaf(child)) {
        // Concantenate prefixes
        unsigned l1 = node->prefixLength;
        if (l1 < maxPrefixLength) {
          node->prefix[l1] = node->key[0];
          l1++;
        }
        if (l1 < maxPrefixLength) {
          unsigned l2 = min(child->prefixLength, maxPrefixLength - l1);
          memcpy(node->prefix + l1, child->prefix, l2);
          l1 += l2;
        }
        // Store concantenated prefix
        memcpy(child->prefix, node->prefix, min(l1, maxPrefixLength));
        child->prefixLength += node->prefixLength + 1;
      }
      *nodeRef = child;
      delete node;
    }
  }

  void eraseNode16(Node16* node, Node** nodeRef, Node** leafPlace) {
    // Delete leaf from inner node
    unsigned pos = leafPlace - node->child;
    memmove(node->key + pos, node->key + pos + 1, node->count - pos - 1);
    memmove(node->child + pos, node->child + pos + 1,
            (node->count - pos - 1) * sizeof(uintptr_t));
    node->count--;

    if (node->count == 3) {
      // Shrink to Node4
      Node4* newNode = new Node4();
      newNode->count = 4;
      copyPrefix(node, newNode);
      for (unsigned i = 0; i < 4; i++) newNode->key[i] = node->key[i];
      memcpy(newNode->child, node->child, sizeof(uintptr_t) * 4);
      *nodeRef = newNode;
      delete node;
    }
  }

  void eraseNode48(Node48* node, Node** nodeRef, uint8_t keyByte) {
    // Delete leaf from inner node
    node->child[node->childIndex[keyByte]] = NULL;
    node->childIndex[keyByte] = emptyMarker;
    node->count--;

    if (node->count == 12) {
      // Shrink to Node16
      Node16* newNode = new Node16();
      *nodeRef = newNode;
      copyPrefix(node, newNode);
      for (unsigned b = 0; b < 256; b++) {
        if (node->childIndex[b] != emptyMarker) {
          newNode->key[newNode->count] = b;
          newNode->child[newNode->count] = node->child[node->childIndex[b]];
          newNode->count++;
        }
      }
      delete node;
    }
  }

  void eraseNode256(Node256* node, Node** nodeRef, uint8_t keyByte) {
    // Delete leaf from inner node
    node->child[keyByte] = NULL;
    node->count--;

    if (node->count == 37) {
      // Shrink to Node48
      Node48* newNode = new Node48();
      *nodeRef = newNode;
      copyPrefix(node, newNode);
      for (unsigned b = 0; b < 256; b++) {
        if (node->child[b]) {
          newNode->childIndex[b] = newNode->count;
          newNode->child[newNode->count] = node->child[b];
          newNode->count++;
        }
      }
      delete node;
    }
  }

  void destructTree(Node* node) {
    if (isLeaf(node)){
      delete reinterpret_cast<Element<std::string>*>(getLeafValue(node) << 1);
      return;
    }

    switch (node->type) {
      case NodeType4: {
        auto n4 = static_cast<Node4*>(node);
        for (auto i = 0; i < node->count; i++) {
          destructTree(n4->child[i]);
        }
        delete n4;
        break;
      }
      case NodeType16: {
        auto n16 = static_cast<Node16*>(node);
        for (auto i = 0; i < node->count; i++) {
          destructTree(n16->child[i]);
        }
        delete n16;
        break;
      }
      case NodeType48: {
        auto n48 = static_cast<Node48*>(node);
        for (auto i = 0; i < 256; i++) {
          if (n48->childIndex[i] != emptyMarker) {
            destructTree(n48->child[n48->childIndex[i]]);
          }
        }
        delete n48;
        break;
      }
      case NodeType256: {
        auto n256 = static_cast<Node256*>(node);
        for (auto i = 0; i < 256; i++) {
          if (n256->child[i] != nullptr) {
            destructTree(n256->child[i]);
          }
        }
        delete n256;
        break;
      }
    }
  }

  uint64_t size_in_bytes(Node* node) const {
    if (isLeaf(node)){
      return loadKey(getLeafValue(node)).length() + sizeof(uint64_t);
    }

    uint64_t size = 0;

    switch (node->type) {
      case NodeType4: {
        auto n4 = static_cast<Node4*>(node);
        for (auto i = 0; i < node->count; i++) {
          size += size_in_bytes(n4->child[i]);
        }
        size += sizeof(*n4);
        break;
      }
      case NodeType16: {
        auto n16 = static_cast<Node16*>(node);
        for (auto i = 0; i < node->count; i++) {
          size += size_in_bytes(n16->child[i]);
        }
        size += sizeof(*n16);
        break;
      }
      case NodeType48: {
        auto n48 = static_cast<Node48*>(node);
        for (auto i = 0; i < 256; i++) {
          if (n48->childIndex[i] != emptyMarker) {
            size += size_in_bytes(n48->child[n48->childIndex[i]]);
          }
        }
        size += sizeof(*n48);
        break;
      }
      case NodeType256: {
        auto n256 = static_cast<Node256*>(node);
        for (auto i = 0; i < 256; i++) {
          if (n256->child[i] != nullptr) {
            size += size_in_bytes(n256->child[i]);
          }
        }
        size += sizeof(*n256);
        break;
      }
    }

    return size;
  }

  Node* tree_ = NULL;
};

};