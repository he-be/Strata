// src/program/token_grammar.hpp - constrained decoding for --serve (response_format json_schema).
//
// The server compiles the schema into a BYTE-level DFA (serve/grammar.py) and writes it to a file; a request names
// it with `grammar=<path>`.  Here the DFA meets the vocabulary: a token may follow state s when all of its bytes
// walk the DFA without dying, and the state after it is where the walk ends.  The token -> bytes table is its own
// file (written once by the server, named inside the grammar file), so a grammar is only the schema's automaton.
//
// Special tokens have no bytes: the end of the answer (the grammar's eos ids) is allowed only in an accepting
// state, and a grammar compiled for a thinking request starts in a FREE state that takes any ordinary token until
// `</think>`, which moves it to the DFA's start.  Every other special token is never allowed.
//
// Masks are computed on first use per state (one walk of the vocabulary, a few ms) and kept with the grammar,
// which is cached by path: the same schema in the next request costs nothing.
//
//   vocab file:   "SVOC" u32 version=1, u32 n, then n x (u8 special, u16 len, len bytes)
//   grammar file: "SGRM" u32 version=1, u32 n_states, i32 start, i32 think_end (-1: none), u32 n_eos,
//                 n_eos x i32, u32 vocab_path_len, the path, n_states x u8 accept, n_states x 256 x i32 next (-1: dead)
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::program {

struct TokenVocab {
    std::vector<uint32_t> off;       // n + 1 offsets into bytes
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> special;

    size_t size() const { return special.size(); }

    static std::shared_ptr<TokenVocab> load(const std::string& path, std::string& err) {
        static std::map<std::string, std::shared_ptr<TokenVocab>> cache;
        if (auto it = cache.find(path); it != cache.end()) return it->second;
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "grammar: cannot open the vocabulary " + path; return nullptr; }
        auto v = std::make_shared<TokenVocab>();
        char magic[4];
        uint32_t ver = 0, n = 0;
        bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "SVOC", 4) == 0 &&
                  std::fread(&ver, 4, 1, f) == 1 && ver == 1 && std::fread(&n, 4, 1, f) == 1;
        v->off.reserve((size_t) n + 1);
        v->special.reserve(n);
        v->off.push_back(0);
        for (uint32_t i = 0; ok && i < n; ++i) {
            uint8_t sp = 0;
            uint16_t len = 0;
            ok = std::fread(&sp, 1, 1, f) == 1 && std::fread(&len, 2, 1, f) == 1;
            if (!ok) break;
            const size_t at = v->bytes.size();
            v->bytes.resize(at + len);
            ok = len == 0 || std::fread(v->bytes.data() + at, 1, len, f) == len;
            v->special.push_back(sp);
            v->off.push_back((uint32_t) v->bytes.size());
        }
        std::fclose(f);
        if (!ok) { err = "grammar: a broken vocabulary file " + path; return nullptr; }
        cache[path] = v;
        return v;
    }
};

class TokenGrammar {
public:
    static constexpr int32_t kFree = -2;   // the thinking part: any ordinary token until </think>

    static std::shared_ptr<TokenGrammar> load(const std::string& path, int64_t n_vocab, std::string& err) {
        static std::map<std::string, std::shared_ptr<TokenGrammar>> cache;   // the server names files by content
        if (auto it = cache.find(path); it != cache.end()) return it->second;
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "grammar: cannot open " + path; return nullptr; }
        auto g = std::make_shared<TokenGrammar>();
        char magic[4];
        uint32_t ver = 0, n_states = 0, n_eos = 0, plen = 0;
        bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "SGRM", 4) == 0 &&
                  std::fread(&ver, 4, 1, f) == 1 && ver == 1 && std::fread(&n_states, 4, 1, f) == 1 &&
                  std::fread(&g->start_, 4, 1, f) == 1 && std::fread(&g->think_end_, 4, 1, f) == 1 &&
                  std::fread(&n_eos, 4, 1, f) == 1 && n_eos < 64;
        if (ok) {
            g->eos_.resize(n_eos);
            ok = n_eos == 0 || std::fread(g->eos_.data(), 4, n_eos, f) == n_eos;
        }
        std::string vpath;
        ok = ok && std::fread(&plen, 4, 1, f) == 1 && plen < 4096;
        if (ok) {
            vpath.resize(plen);
            ok = plen == 0 || std::fread(vpath.data(), 1, plen, f) == plen;
        }
        if (ok) {
            g->accept_.resize(n_states);
            g->next_.resize((size_t) n_states * 256);
            ok = std::fread(g->accept_.data(), 1, n_states, f) == n_states &&
                 std::fread(g->next_.data(), 4, g->next_.size(), f) == g->next_.size();
        }
        std::fclose(f);
        if (!ok || n_states == 0 || g->start_ < 0 || g->start_ >= (int32_t) n_states) {
            err = "grammar: a broken grammar file " + path;
            return nullptr;
        }
        g->vocab_ = TokenVocab::load(vpath, err);
        if (!g->vocab_) return nullptr;
        g->n_states_ = (int32_t) n_states;
        g->words_ = (int) ((n_vocab + 31) / 32);
        g->masks_.resize(n_states);
        cache[path] = g;
        return g;
    }

    int words() const { return words_; }
    int32_t initial() const { return think_end_ >= 0 ? kFree : start_; }

    /// The state after `tok` from `s`, or -1 when `tok` may not follow `s`.
    int32_t step(int32_t s, int32_t tok) const {
        if (s == -1 || tok < 0 || (size_t) tok >= vocab_->size()) return -1;
        if (vocab_->special[(size_t) tok]) {
            if (s == kFree) return tok == think_end_ ? start_ : -1;
            return accept_[(size_t) s] && is_eos(tok) ? s : -1;
        }
        if (s == kFree) return kFree;
        const uint32_t b = vocab_->off[(size_t) tok], e = vocab_->off[(size_t) tok + 1];
        if (b == e) return -1;
        for (uint32_t i = b; i < e && s >= 0; ++i) s = next_[(size_t) s * 256 + vocab_->bytes[i]];
        return s;
    }

    /// The allowed tokens after `s`, `words()` uint32.
    const std::vector<uint32_t>& mask(int32_t s) {
        std::vector<uint32_t>& m = s == kFree ? free_mask_ : masks_[(size_t) s];
        if (!m.empty()) return m;
        m.assign((size_t) words_, 0u);
        const int64_t n = std::min<int64_t>((int64_t) vocab_->size(), (int64_t) words_ * 32);
        for (int64_t t = 0; t < n; ++t)
            if (step(s, (int32_t) t) != -1) m[(size_t) t >> 5] |= 1u << (t & 31);
        return m;
    }

private:
    bool is_eos(int32_t t) const {
        for (int32_t e : eos_) if (e == t) return true;
        return false;
    }

    std::shared_ptr<TokenVocab> vocab_;
    int32_t n_states_ = 0, start_ = 0, think_end_ = -1;
    int words_ = 0;
    std::vector<int32_t> eos_;
    std::vector<uint8_t> accept_;
    std::vector<int32_t> next_;
    std::vector<std::vector<uint32_t>> masks_;
    std::vector<uint32_t> free_mask_;
};

}  // namespace strata::program
