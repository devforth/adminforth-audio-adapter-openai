import OpenAI, { toFile } from "openai";
import type {
  OpenAITtsVoice,
  SpeechToTextAdapter,
  SpeechToTextInput,
  SpeechToTextResult,
  TextToSpeechAdapter,
  TextToSpeechInput,
  TextToSpeechResult,
  TextToSpeechStreamInput,
  TextToSpeechStreamResult,
  TtsAudioFormat,
} from "./audioAdapters.js";
//redeploy
export type OpenAITranscriptionModel =
  | "gpt-4o-transcribe"
  | "gpt-4o-mini-transcribe"
  | "whisper-1";

export type OpenAISpeechModel =
  | "gpt-4o-mini-tts"
  | "tts-1"
  | "tts-1-hd";

export type OpenAIAudioAdapterOptions = {
  client?: OpenAI;
  apiKey?: string;
  transcriptionModel?: OpenAITranscriptionModel;
  speechModel?: OpenAISpeechModel;
  defaultVoice?: OpenAITtsVoice;
  defaultAudioFormat?: TtsAudioFormat;
  maxAudioFileSizeBytes?: number;
};

const DEFAULT_MAX_AUDIO_FILE_SIZE_BYTES = 25 * 1024 * 1024;

const AUDIO_MIME_TYPES = new Set([
  "audio/mpeg",
  "audio/mp3",
  "audio/mp4",
  "audio/mpga",
  "audio/m4a",
  "audio/wav",
  "audio/wave",
  "audio/webm",
  "audio/ogg",
  "audio/flac",
  "video/mp4",
  "video/webm",
]);

const MIME_TYPE_BY_TTS_FORMAT: Record<TtsAudioFormat, string> = {
  mp3: "audio/mpeg",
  opus: "audio/opus",
  aac: "audio/aac",
  flac: "audio/flac",
  wav: "audio/wav",
  pcm: "audio/pcm",
};

export class OpenAIAudioAdapter
  implements SpeechToTextAdapter, TextToSpeechAdapter
{
  public readonly name = "openai";

  private client?: OpenAI;
  private readonly transcriptionModel: OpenAITranscriptionModel;
  private readonly speechModel: OpenAISpeechModel;
  private readonly defaultVoice: OpenAITtsVoice;
  private readonly defaultAudioFormat: TtsAudioFormat;
  private readonly maxAudioFileSizeBytes: number;
  private readonly apiKey?: string;
  private readonly hasCustomClient: boolean;

  constructor(options: OpenAIAudioAdapterOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    this.hasCustomClient = Boolean(options.client);
    this.client = options.client;
    this.transcriptionModel =
      options.transcriptionModel ?? "gpt-4o-mini-transcribe";
    this.speechModel = options.speechModel ?? "gpt-4o-mini-tts";
    this.defaultVoice = options.defaultVoice ?? "coral";
    this.defaultAudioFormat = options.defaultAudioFormat ?? "mp3";
    this.maxAudioFileSizeBytes =
      options.maxAudioFileSizeBytes ?? DEFAULT_MAX_AUDIO_FILE_SIZE_BYTES;
  }

  validate(): void {
    if (!this.hasCustomClient && !this.apiKey) {
      throw new Error("OpenAI API key is required");
    }
  }

  private getClient(): OpenAI {
    this.validate();
    this.client ??= new OpenAI({
      apiKey: this.apiKey,
    });
    return this.client;
  }

  async transcribe(input: SpeechToTextInput): Promise<SpeechToTextResult> {
    this.validateAudioInput(input);

    const file = await toFile(input.buffer, input.filename, {
      type: input.mimeType,
    });
    const language = input.language === "auto" ? undefined : input.language;
    const result = await this.getClient().audio.transcriptions.create({
      model: this.transcriptionModel,
      file,
      language,
      prompt: input.prompt,
      response_format: "json",
    });

    return {
      text: result.text.trim(),
      raw: result,
    };
  }

  async synthesize(input: TextToSpeechStreamInput): Promise<TextToSpeechStreamResult>;
  async synthesize(input: TextToSpeechInput): Promise<TextToSpeechResult>;
  async synthesize(
    input: TextToSpeechInput | TextToSpeechStreamInput,
  ): Promise<TextToSpeechResult | TextToSpeechStreamResult> {
    const text = input.text.trim();

    if (!text) {
      throw new Error("TTS input text is empty");
    }

    const format = input.format ?? this.defaultAudioFormat;
    const streamFormat = input.stream ? input.streamFormat ?? "audio" : undefined;
    const response = await this.getClient().audio.speech.create({
      model: this.speechModel,
      voice: input.voice ?? this.defaultVoice,
      input: text,
      response_format: format,
      speed: input.speed,
      instructions: input.instructions,
      stream_format: streamFormat,
    });

    if (input.stream) {
      const selectedStreamFormat = streamFormat ?? "audio";

      if (!response.body) {
        throw new Error("TTS stream response body is empty");
      }

      return {
        audioStream: response.body,
        mimeType:
          selectedStreamFormat === "sse" ? "text/event-stream" : MIME_TYPE_BY_TTS_FORMAT[format],
        format,
        streamFormat: selectedStreamFormat,
        raw: response,
      };
    }

    const audio = Buffer.from(await response.arrayBuffer());

    return {
      audio,
      mimeType: MIME_TYPE_BY_TTS_FORMAT[format],
      format,
      raw: response,
    };
  }

  private validateAudioInput(input: SpeechToTextInput): void {
    if (!input.buffer.length) {
      throw new Error("Audio buffer is empty");
    }

    if (input.buffer.length > this.maxAudioFileSizeBytes) {
      throw new Error(
        `Audio file is too large. Maximum size is ${this.maxAudioFileSizeBytes} bytes`,
      );
    }

    if (!input.filename) {
      throw new Error("Audio filename is required");
    }

    if (!input.mimeType) {
      throw new Error("Audio MIME type is required");
    }

    if (!AUDIO_MIME_TYPES.has(input.mimeType)) {
      throw new Error(`Unsupported audio MIME type: ${input.mimeType}`);
    }
  }
}

export default OpenAIAudioAdapter;
