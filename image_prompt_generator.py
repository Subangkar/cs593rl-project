"""
Image-based prompt generation module for creating typographic images from text prompts.
Suitable for VLM inputs via Ollama.
"""

from PIL import Image, ImageFont, ImageDraw
from enum import IntEnum, unique
import textwrap
import os
from pathlib import Path


@unique
class ImagePromptStyle(IntEnum):
    """Different styles for image-based prompts"""
    simple_text = 0           # Plain text on white background
    stepwise = 1              # Numbered steps (1, 2, 3, etc.)
    archaic_english = 2       # Formal/archaic style text
    technical_jargon = 3      # Technical presentation
    highlighted = 4           # Highlighted key terms
    multi_line = 5            # Multi-line formatted text


class TextToImageConverter:
    """Convert text prompts to typographic images for VLM processing"""
    
    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        bg_color: str = "#FFFFFF",
        text_color: str = "#000000",
        font_size: int = 60,
        font_path: str = "FreeMonoBold.ttf",
        output_dir: str = "./temp_images",
    ):
        """
        Initialize the text-to-image converter.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            bg_color: Background color (hex)
            text_color: Text color (hex)
            font_size: Font size for text
            font_path: Path to font file
            output_dir: Directory to save generated images
        """
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.text_color = text_color
        self.font_size = font_size
        self.font_path = font_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load font
        try:
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        except (IOError, OSError):
            # Fallback to default font if custom font not found
            print(f"Warning: Font {self.font_path} not found, using default font")
            self.font = ImageFont.load_default()
    
    def _get_text_bbox(self, text: str, xy: tuple = (20, 10)) -> tuple:
        """Get bounding box for text to calculate image size"""
        im = Image.new("RGB", (0, 0))
        dr = ImageDraw.Draw(im)
        return dr.textbbox(xy=xy, text=text, font=self.font)
    
    def _wrap_text(self, text: str, width: int = 30) -> str:
        """Wrap text to specified width"""
        return textwrap.fill(text, width=width)
    
    def _calculate_required_height(self, text: str, margin: int = 40) -> int:
        """Calculate required image height based on text content"""
        # Create temporary image for measurement
        temp_im = Image.new("RGB", (self.width, 100))
        temp_dr = ImageDraw.Draw(temp_im)
        
        # Get text bounding box
        bbox = temp_dr.textbbox(xy=(20, 20), text=text, font=self.font, spacing=10)
        text_height = bbox[3] - bbox[1]
        
        # Add margins and ensure minimum height
        required_height = text_height + margin
        return max(required_height, 200)  # Minimum 200px
    
    def simple_text_image(
        self,
        text: str,
        filename: str = "prompt_simple.png",
        wrap: bool = True,
        wrap_width: int = 30,
        auto_height: bool = True,
    ) -> str:
        """
        Create a simple text image with automatic height adjustment.
        
        Args:
            text: Text to render
            filename: Output filename
            wrap: Whether to wrap text
            wrap_width: Character width for wrapping
            auto_height: Automatically adjust height to fit text
            
        Returns:
            Path to saved image
        """
        if wrap:
            text = self._wrap_text(text, wrap_width)
        
        # Calculate required height if auto_height is enabled
        if auto_height:
            height = self._calculate_required_height(text)
            # Cap maximum height to prevent extremely large images
            height = min(height, 4096)
        else:
            height = self.height
        
        im = Image.new("RGB", (self.width, height), self.bg_color)
        dr = ImageDraw.Draw(im)
        
        dr.text(
            xy=(20, 20),
            text=text,
            fill=self.text_color,
            font=self.font,
            spacing=10,
        )
        
        output_path = os.path.join(self.output_dir, filename)
        im.save(output_path)
        return output_path
    
    def stepwise_image(
        self,
        text: str,
        steps: int = 3,
        filename: str = "prompt_stepwise.png",
        wrap: bool = True,
        auto_height: bool = True,
    ) -> str:
        """
        Create a stepwise numbered image (1, 2, 3, ...).
        Useful for prompting VLMs to fill in numbered steps.
        
        Args:
            text: Instruction text
            steps: Number of steps to generate
            filename: Output filename
            wrap: Whether to wrap text
            auto_height: Automatically adjust height to fit text
            
        Returns:
            Path to saved image
        """
        if wrap:
            text = self._wrap_text(text, width=20)
        
        # Add numbered steps
        full_text = text.rstrip("\n")
        for idx in range(1, steps + 1):
            full_text += f"\n{idx}. "
        
        # Calculate required height if auto_height is enabled
        if auto_height:
            height = self._calculate_required_height(full_text)
            height = min(height, 4096)
        else:
            height = self.height
        
        im = Image.new("RGB", (self.width, height), self.bg_color)
        dr = ImageDraw.Draw(im)
        
        dr.text(
            xy=(20, 20),
            text=full_text,
            fill=self.text_color,
            font=self.font,
            spacing=10,
        )
        
        output_path = os.path.join(self.output_dir, filename)
        im.save(output_path)
        return output_path
    
    def archaic_english_image(
        self,
        text: str,
        filename: str = "prompt_archaic.png",
        wrap: bool = True,
        auto_height: bool = True,
    ) -> str:
        """
        Create an image with archaic/old English style formatting.
        
        Args:
            text: Text to render
            filename: Output filename
            wrap: Whether to wrap text
            auto_height: Automatically adjust height to fit text
            
        Returns:
            Path to saved image
        """
        # Add some archaic styling markers
        styled_text = f"~ {text} ~\n\n[Request in the manner of old English]"
        
        if wrap:
            styled_text = self._wrap_text(styled_text, width=25)
        
        # Calculate required height if auto_height is enabled
        if auto_height:
            height = self._calculate_required_height(styled_text)
            height = min(height, 4096)
        else:
            height = self.height
        
        im = Image.new("RGB", (self.width, height), self.bg_color)
        dr = ImageDraw.Draw(im)
        
        dr.text(
            xy=(20, 20),
            text=styled_text,
            fill=self.text_color,
            font=self.font,
            spacing=10,
        )
        
        output_path = os.path.join(self.output_dir, filename)
        im.save(output_path)
        return output_path
    
    def technical_jargon_image(
        self,
        text: str,
        filename: str = "prompt_technical.png",
        wrap: bool = True,
        auto_height: bool = True,
    ) -> str:
        """
        Create an image with technical/formal presentation.
        
        Args:
            text: Text to render
            filename: Output filename
            wrap: Whether to wrap text
            auto_height: Automatically adjust height to fit text
            
        Returns:
            Path to saved image
        """
        styled_text = f"[TECHNICAL SPECIFICATION]\n\n{text}\n\n[END SPECIFICATION]"
        
        if wrap:
            styled_text = self._wrap_text(styled_text, width=25)
        
        # Calculate required height if auto_height is enabled
        if auto_height:
            height = self._calculate_required_height(styled_text)
            height = min(height, 4096)
        else:
            height = self.height
        
        im = Image.new("RGB", (self.width, height), self.bg_color)
        dr = ImageDraw.Draw(im)
        
        dr.text(
            xy=(20, 20),
            text=styled_text,
            fill=self.text_color,
            font=self.font,
            spacing=10,
        )
        
        output_path = os.path.join(self.output_dir, filename)
        im.save(output_path)
        return output_path
    
    def highlighted_image(
        self,
        text: str,
        highlight_words: list = None,
        filename: str = "prompt_highlighted.png",
        wrap: bool = True,
        auto_height: bool = True,
    ) -> str:
        """
        Create an image with highlighted key terms.
        
        Args:
            text: Text to render
            highlight_words: List of words to highlight (uppercase)
            filename: Output filename
            wrap: Whether to wrap text
            auto_height: Automatically adjust height to fit text
            
        Returns:
            Path to saved image
        """
        if highlight_words:
            for word in highlight_words:
                text = text.replace(word, word.upper())
        
        if wrap:
            text = self._wrap_text(text, width=25)
        
        # Calculate required height if auto_height is enabled
        if auto_height:
            height = self._calculate_required_height(text)
            height = min(height, 4096)
        else:
            height = self.height
        
        im = Image.new("RGB", (self.width, height), self.bg_color)
        dr = ImageDraw.Draw(im)
        
        dr.text(
            xy=(20, 20),
            text=text,
            fill=self.text_color,
            font=self.font,
            spacing=10,
        )
        
        output_path = os.path.join(self.output_dir, filename)
        im.save(output_path)
        return output_path
    
    def multi_line_image(
        self,
        text: str,
        lines_per_section: int = 3,
        filename: str = "prompt_multiline.png",
        auto_height: bool = True,
    ) -> str:
        """
        Create an image with multi-line formatted text.
        
        Args:
            text: Text to render
            lines_per_section: Lines per section
            filename: Output filename
            auto_height: Automatically adjust height to fit text
            
        Returns:
            Path to saved image
        """
        # Split text into sentences and reformat
        sentences = text.split(". ")
        formatted_lines = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                formatted_lines.append(f"• {sentence.strip()}")
                if (i + 1) % lines_per_section == 0:
                    formatted_lines.append("")
        
        formatted_text = "\n".join(formatted_lines)
        
        # Calculate required height if auto_height is enabled
        if auto_height:
            height = self._calculate_required_height(formatted_text)
            height = min(height, 4096)
        else:
            height = self.height
        
        im = Image.new("RGB", (self.width, height), self.bg_color)
        dr = ImageDraw.Draw(im)
        
        dr.text(
            xy=(20, 20),
            text=formatted_text,
            fill=self.text_color,
            font=self.font,
            spacing=10,
        )
        
        output_path = os.path.join(self.output_dir, filename)
        im.save(output_path)
        return output_path
    
    def generate_image(
        self,
        text: str,
        style: ImagePromptStyle = ImagePromptStyle.simple_text,
        filename: str = None,
        **kwargs
    ) -> str:
        """
        Generate image based on style.
        
        Args:
            text: Text to render
            style: ImagePromptStyle enum value
            filename: Output filename (auto-generated if None)
            **kwargs: Style-specific keyword arguments
            
        Returns:
            Path to saved image
        """
        if filename is None:
            filename = f"prompt_{style.name}.png"
        
        if style == ImagePromptStyle.simple_text:
            return self.simple_text_image(text, filename, **kwargs)
        elif style == ImagePromptStyle.stepwise:
            return self.stepwise_image(text, filename=filename, **kwargs)
        elif style == ImagePromptStyle.archaic_english:
            return self.archaic_english_image(text, filename, **kwargs)
        elif style == ImagePromptStyle.technical_jargon:
            return self.technical_jargon_image(text, filename, **kwargs)
        elif style == ImagePromptStyle.highlighted:
            return self.highlighted_image(text, filename=filename, **kwargs)
        elif style == ImagePromptStyle.multi_line:
            return self.multi_line_image(text, filename=filename, **kwargs)
        else:
            raise ValueError(f"Unknown style: {style}")
