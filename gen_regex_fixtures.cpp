// Writes src/test_fixtures/regex_cases.txt: what std::regex_search (ECMAScript,
// the engine llama-quantize matches --tensor-type patterns with) answers for
// each pattern against each subject. Regex.zig is tested against it.
//
//   g++ -std=c++17 -O1 gen_regex_fixtures.cpp -o /tmp/gen_regex && /tmp/gen_regex > src/test_fixtures/regex_cases.txt
//
// Line format: pattern<TAB>subject<TAB>0|1, or pattern<TAB><TAB>ERR when
// std::regex refuses the pattern. Patterns Regex.zig refuses on purpose
// (lookaround, backreferences, collating elements, \cX) are left out.

#include <cstdio>
#include <regex>
#include <string>
#include <vector>

int main() {
    const std::vector<std::string> patterns = {
        // Shapes seen in published llama-quantize recipes.
        "attn_v", "attn_q", "ffn_down", "token_embd", "output", "output\\.weight",
        "blk\\.0\\.", "blk\\.1[0-9]\\.", "blk\\.[0-9]+\\.ffn_down", "blk\\.(0|1|2)\\.attn",
        "blk\\.(?:0|1|2)\\.attn", "ffn_.*_exps", "ffn_(up|gate)_exps", "\\.ffn_down_exps\\.",
        "blk\\.([0-9]|1[0-5])\\.ffn", "blk\\.\\d+\\.attn_(q|k|v)\\.weight", "^output\\.weight$",
        "^blk\\.0\\.", "_exps$", "weight$", "^token", "attn_.*", "blk\\.[2-3][0-9]\\.ffn_up",
        "blk\\.(1[6-9]|2[0-9]|3[01])\\.", "ffn_(down|up)_shexp", "attn_(kv_a_mqa|q_a|q_b|kv_b)",
        "blk\\.[0-9]{1,2}\\.ffn_gate", "blk\\.\\d{2}\\.", "blk\\.\\d{1}\\.", "blk\\.\\d{1,}\\.",
        "blk\\.[^0]\\.", "[.]ffn", "\\bffn", "ffn\\b", "\\bblk\\b", "blk\\B",
        "double_blocks\\.[0-9]+\\.img_attn\\.qkv", "single_blocks\\.(0|1|2)\\.", "img_mlp\\.0",
        "transformer_blocks\\.\\d+\\.attn\\d?\\.to_[qkv]",
        // Syntax edges, including libstdc++'s own leniencies.
        "a", "a|b", "a|", "|a", "()", "(a)(b)", "a*", "a+", "a?", "a**", "a*?", "a+?", "a??",
        "a{2}", "a{2,}", "a{1,3}", "a{0,0}", "(ab){2}", "(a|bc)+d", "x(a|)y", "[abc]", "[^abc]",
        "[a-c]", "[-a]", "[a-]", "[]", "[^]", "[]a]", "[\\]]", "[\\-]", "[\\d]", "[^\\d]",
        "[\\w.]", "[\\s]", "[\\b]", "[[:digit:]]", "[[:alpha:]_]", "[^[:alnum:]]", "\\d", "\\D",
        "\\w+", "\\W", "\\s", "\\S", "\\.", ".", "..", "^$", "^", "$", "a$", "^a", "\\^", "\\$",
        "}", "a}", "]", "\\q", "\\_", "\\-", "\\/", "\\x41", "\\x61", "\\u0061", "\\t",
        "(a(b(c)))", "((a|b)c|d)+", "(a*)*", "(a*)+b", "(?:a|b)*c", "a.c", "a.*c", "a.+c",
        "a[0-9]*b", "[0-9]+$", "^[a-z_.0-9]+$",
        // Refused by std::regex.
        "a{", "a{2", "a{3,2}", "x{,2}", "{2}", "^*", "$+", "\\b*", "(", "a)", "[a", "[a-\\d]",
        "[\\d-z]", "[z-a]", "*a", "+a", "?a", "\\x4", "\\u004",
    };
    const std::vector<std::string> subjects = {
        "", "a", "aa", "aaa", "b", "ab", "abc", "abab", "ac", "abcabcd", "xy", "xay", "xby",
        "a1234b", "ab}", "a}", "]", "-", "_", "q", "/", "\x01", ".", "1", "12", "x.y",
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.attn_norm.weight",
        "blk.1.ffn_down.weight", "blk.5.ffn_gate.weight", "blk.12.ffn_down.weight",
        "blk.15.ffn_up.weight", "blk.17.attn_k.weight", "blk.23.ffn_up.weight",
        "blk.30.ffn_down_exps.weight", "blk.31.ffn_gate_exps.weight", "blk.3.ffn_up_exps.weight",
        "blk.4.ffn_down_shexp.weight", "blk.2.attn_kv_a_mqa.weight", "blk.40.attn_output.weight",
        "double_blocks.0.img_attn.qkv.weight", "double_blocks.18.img_mlp.0.weight",
        "single_blocks.1.linear1.weight", "single_blocks.37.linear2.weight",
        "transformer_blocks.3.attn.to_q.weight", "transformer_blocks.10.attn2.to_k.weight",
        "model.diffusion_model.input_blocks.4.1.proj_in.weight",
    };

    for (const auto & p : patterns) {
        std::regex re;
        try {
            re = std::regex(p, std::regex::ECMAScript);
        } catch (const std::regex_error &) {
            std::printf("%s\t\tERR\n", p.c_str());
            continue;
        }
        for (const auto & s : subjects) {
            std::printf("%s\t%s\t%d\n", p.c_str(), s.c_str(), std::regex_search(s, re) ? 1 : 0);
        }
    }
}
