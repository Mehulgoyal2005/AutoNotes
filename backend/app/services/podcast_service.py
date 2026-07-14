import json
import httpx
from pathlib import Path
from typing import Tuple, Optional
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup
import random


class PodcastService:
    """Service for generating podcast audio from document content"""

    def __init__(
        self,
        vector_store,
        audio_output_dir: Path,
        groq_api_key: str = None,
        max_duration: int = 180
    ):
        self.vector_store = vector_store
        self.audio_output_dir = audio_output_dir
        self.groq_api_key = groq_api_key
        self.max_duration = max_duration

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to the .env file at the project root."
            )

        print(f"🎙️  Podcast Service initialized")
        print(f"   - TTS: Google TTS (gTTS)")
        print(f"   - LLM: Groq API")

    def _query_llm(self, prompt: str) -> str:
        """Query Groq LLM for dialogue generation"""
        try:
            print("🤖 Using Groq API for dialogue generation...")
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 2500,
                        "response_format": {"type": "json_object"}  # Force JSON output
                    }
                )
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                    print(f"✅ Successfully received response from Groq API")
                    return result
                else:
                    print(f"⚠️  Groq API returned status {response.status_code}: {response.text}")
                    return ""
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return ""

    def generate_dialogue(self, document_id: str) -> list:
        """Generate podcast dialogue from document content"""
        # Get document chunks
        chunks = self.vector_store.get_all_chunks(document_id)
        if not chunks:
            return []

        # Combine chunks for context (limit to avoid token overflow)
        content = "\n".join(chunks[:10])[:6000]

        # Create improved prompt for more natural dialogue
        prompt = f"""Create a podcast script as a JSON array. Two hosts discuss a document naturally.

HOSTS:
- Alex: Enthusiastic, curious, asks questions
- Jordan: Knowledgeable, explains clearly

RULES:
1. NO page numbers, URLs, or citations
2. Use contractions and casual language
3. Natural reactions: "Oh wow", "That's fascinating"
4. Make 6-8 exchanges total

DOCUMENT:
{content}

OUTPUT FORMAT (JSON only, no other text):
{{
  "dialogue": [
    {{"speaker": "Alex", "text": "Hey everyone! Welcome back..."}},
    {{"speaker": "Jordan", "text": "Thanks Alex! Today we're exploring..."}}
  ]
}}"""

        # Get dialogue from LLM
        response = self._query_llm(prompt)

        if not response:
            print("❌ No response from LLM")
            return self._get_fallback_dialogue()

        # Try to parse JSON response
        try:
            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code fences
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response

            # Try to parse as structured JSON first
            try:
                parsed = json.loads(response)
                # Check if it has a 'dialogue' key
                if isinstance(parsed, dict) and 'dialogue' in parsed:
                    dialogue = parsed['dialogue']
                    print(f"✅ Generated {len(dialogue)} dialogue exchanges from structured JSON")
                    return dialogue
                # If it's already an array, use it directly
                elif isinstance(parsed, list):
                    print(f"✅ Generated {len(parsed)} dialogue exchanges from array JSON")
                    return parsed
            except json.JSONDecodeError:
                # If full parse fails, try to extract JSON array
                start = response.find('[')
                end = response.rfind(']') + 1
                if start != -1 and end > start:
                    dialogue_json = response[start:end]
                    dialogue = json.loads(dialogue_json)
                    print(f"✅ Generated {len(dialogue)} dialogue exchanges (extracted from response)")
                    return dialogue
                else:
                    raise ValueError("No JSON array found in response")

        except Exception as e:
            print(f"⚠️  Error parsing dialogue JSON: {e}")
            print(f"📝 Response preview: {response[:200]}...")
            return self._get_fallback_dialogue()

    def _get_fallback_dialogue(self) -> list:
        """Return fallback dialogue template"""

        # Fallback: create natural dialogue from content
        print("ℹ️  Using fallback dialogue template")
        return [
            {"speaker": "Alex", "text": "Hey everyone! Welcome back. Today we've got something really interesting to dive into, and I'm excited to unpack it with you, Jordan."},
            {"speaker": "Jordan", "text": "Thanks Alex! Yeah, this is fascinating stuff. So the main thing we're looking at here is really about understanding the core concepts and how they apply in real-world scenarios."},
            {"speaker": "Alex", "text": "Oh that's interesting! Can you break that down a bit more? What makes this approach different?"},
            {"speaker": "Jordan", "text": "Great question! Think of it this way - instead of just throwing information at you, this really helps you connect the dots and see the bigger picture. It's about making those meaningful connections."},
            {"speaker": "Alex", "text": "I love that! And I bet our listeners are already thinking about how they could use this in their own projects or work."},
            {"speaker": "Jordan", "text": "Exactly! That's what makes it so practical. You can actually take these ideas and apply them right away, which is always the goal, right?"},
            {"speaker": "Alex", "text": "Absolutely! This has been such a great conversation. Any final thoughts you want to leave people with?"},
            {"speaker": "Jordan", "text": "Just remember - the key is to start small, experiment, and build from there. Don't be intimidated by the complexity. Take it step by step."},
            {"speaker": "Alex", "text": "Perfect advice! Thanks so much for breaking this down with me today, Jordan. And to everyone listening, we hope this sparked some ideas for you!"}
        ]

    def text_to_speech(self, text: str, speaker: str) -> AudioSegment:
        """Convert text to speech using gTTS"""
        print(f"🎤 Generating speech for {speaker} using gTTS")
        tts = gTTS(text=text, lang='en', slow=False)

        # Save to temporary file
        temp_file = self.audio_output_dir / f"temp_{speaker}_{random.randint(1000, 9999)}.mp3"
        try:
            tts.save(str(temp_file))
            audio = AudioSegment.from_mp3(str(temp_file))
        finally:
            temp_file.unlink(missing_ok=True)

        # Apply slight modifications for different speakers
        if speaker == "Jordan":
            # Slightly slower and lower pitch
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * 0.95)
            })
            audio = audio.set_frame_rate(44100)
        else:
            # Alex: slightly faster
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * 1.05)
            })
            audio = audio.set_frame_rate(44100)

        return audio

    def create_podcast(
        self,
        document_id: str,
        target_duration: Optional[int] = None
    ) -> Tuple[str, float, str]:
        """
        Create podcast audio from document.
        Returns (audio_file_path, duration, transcript).
        """
        if target_duration is None:
            target_duration = self.max_duration

        # Generate dialogue
        dialogue = self.generate_dialogue(document_id)

        # Convert to audio
        podcast_audio = AudioSegment.silent(duration=500)  # Start with 0.5s silence
        transcript_parts = []

        for item in dialogue:
            speaker = item.get("speaker", "Alex")
            text = item.get("text", "")

            if not text:
                continue

            # Add to transcript
            transcript_parts.append(f"{speaker}: {text}")

            # Convert to speech
            speech_audio = self.text_to_speech(text, speaker)

            # Add speech with small pause
            podcast_audio += speech_audio + AudioSegment.silent(duration=800)

        # Add ending silence
        podcast_audio += AudioSegment.silent(duration=500)

        # Adjust speed if needed to fit target duration
        current_duration = len(podcast_audio) / 1000  # Convert to seconds
        if current_duration > target_duration:
            speed_factor = current_duration / target_duration
            # Limit speed increase to 1.3x for natural sound
            speed_factor = min(speed_factor, 1.3)
            podcast_audio = speedup(podcast_audio, playback_speed=speed_factor)

        # Save final audio
        output_filename = f"podcast_{document_id}.mp3"
        output_path = self.audio_output_dir / output_filename

        podcast_audio.export(str(output_path), format="mp3", bitrate="192k")

        # Get final duration
        final_duration = len(podcast_audio) / 1000

        # Create transcript
        transcript = "\n\n".join(transcript_parts)

        return str(output_path), final_duration, transcript
