#pragma once

// LFM3-Audio prompts: the ASR, TTS and chat templates, the TTS voices and
// ASR language names, and the generation mode a prompt starts in.

#include "engine/community_models/lfm3_audio/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::community_models::lfm3_audio {

// The encoder output fills the <|audio|> ids in order; their count must equal the audio tokens (checked by backbone).
struct Lfm3Prompt {
    std::vector<int32_t> ids;                  // full sequence incl. BOS from the template
    std::vector<std::vector<int32_t>> frames;  // codes for each <|audio_out|> id, in order (history replay)
};

struct Lfm3Voice {
    std::string id, language_name /*"American English"*/, gender /*"female"*/;
};

const std::vector<Lfm3Voice> & lfm3_tts_voices();  // 38 voices
const Lfm3Voice & lfm3_tts_voice(const std::string & voice_id, const std::string & language);  // empty id: language default, female
std::string lfm3_tts_system_prompt(const Lfm3Voice & voice);       // "Perform TTS in American English using a female voice."
std::string lfm3_asr_system_prompt(const std::string & language);  // "auto" -> "Perform ASR."; ISO -> "Perform ASR in <Name>."
inline constexpr const char * kLfm3S2sSystemPrompt = "Respond with interleaved text and audio.";

// Request context appended to a built-in system prompt. The context's outer whitespace is trimmed and it
// follows the built-in after one space; an empty context leaves the built-in unchanged. The byte cap only
// keeps tokenization bounded: a prompt that leaves no room in the 8192-step conversation is rejected later.
// LFM3-Audio follows short, task-shaped context best; long personas and contradictory constraints usually hurt.
inline constexpr size_t kLfm3MaxSystemPromptBytes = 64 * 1024;
std::string append_lfm3_system_prompt(const std::string & built_in, const std::string & context);

// Each builder renders the chat_template text and tokenizes it with encode().
Lfm3Prompt make_lfm3_asr_prompt(const Lfm3Tokenizer &, const std::string & system_prompt, int64_t audio_tokens);
Lfm3Prompt make_lfm3_chat_prompt(const Lfm3Tokenizer &, int64_t audio_tokens, bool thinking, const std::string & system_prompt = {});  //No "<think></think>" or "<think>"
Lfm3Prompt make_lfm3_tts_prompt(const Lfm3Tokenizer &, const std::string & system_prompt, const std::string & text);  // ends <|audio_start|>
// S2S is in chat.h (history).
// InvalidRequestError if ids tokenized from request text hold <|audio|> or <|audio_out|>, which only audio fills.
void require_lfm3_no_audio_placeholders(const std::vector<int32_t> & ids, const Lfm3TokenIds & token_ids);

enum class Lfm3Mode { Text, Audio, Interleaved };
Lfm3Mode lfm3_initial_mode(const Lfm3Prompt & prompt, const Lfm3TokenIds & ids, int32_t end_of_audio);

}  // namespace engine::community_models::lfm3_audio
