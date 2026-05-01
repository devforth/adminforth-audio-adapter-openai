export type SpeechToTextInput = {
  buffer: Buffer;
  filename: string;
  mimeType: string;
  language?: string;
  prompt?: string;
};

export type SpeechToTextResult = {
  text: string;
  language?: string;
  raw?: unknown;
};

export interface SpeechToTextAdapter {
  name: string;

  transcribe(input: SpeechToTextInput): Promise<SpeechToTextResult>;
}

export type OpenAITtsVoice =
  | "alloy"
  | "ash"
  | "ballad"
  | "coral"
  | "echo"
  | "fable"
  | "onyx"
  | "nova"
  | "sage"
  | "shimmer"
  | "verse"
  | "marin"
  | "cedar";

export type TtsAudioFormat =
  | "mp3"
  | "opus"
  | "aac"
  | "flac"
  | "wav"
  | "pcm";

export type TextToSpeechInput = {
  text: string;
  voice?: OpenAITtsVoice;
  format?: TtsAudioFormat;
  speed?: number;
  instructions?: string;
  stream?: false;
};

export type TextToSpeechResult = {
  audio: Buffer;
  mimeType: string;
  format: TtsAudioFormat;
  raw?: unknown;
};

export type TtsStreamFormat = "audio" | "sse";

export type TextToSpeechStreamInput = Omit<TextToSpeechInput, "stream"> & {
  stream: true;
  streamFormat?: TtsStreamFormat;
};

export type TextToSpeechStreamResult = {
  audioStream: ReadableStream<Uint8Array>;
  mimeType: string;
  format: TtsAudioFormat;
  streamFormat: TtsStreamFormat;
  raw?: unknown;
};

export interface TextToSpeechAdapter {
  name: string;

  synthesize(input: TextToSpeechStreamInput): Promise<TextToSpeechStreamResult>;
  synthesize(input: TextToSpeechInput): Promise<TextToSpeechResult>;
}
