"""
================================================================================
 ██████╗ ███╗   ███╗███╗   ██╗██╗ ██████╗██╗██████╗ ███████╗
██╔═══██╗████╗ ████║████╗  ██║██║██╔════╝██║██╔══██╗██╔════╝
██║   ██║██╔████╔██║██╔██╗ ██║██║██║     ██║██║  ██║█████╗
██║   ██║██║╚██╔╝██║██║╚██╗██║██║██║     ██║██║  ██║██╔══╝
╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║╚██████╗██║██████╔╝███████╗
 ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝╚═╝╚═════╝ ╚══════╝

 OMNICIDE CLI — Interactive Terminal Application
 A Transformer Built From Absolute Zero
================================================================================
"""
import os
import sys
import time
import math
import io
import numpy as np

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ==============================================================================
# ANSI COLOR CODES
# ==============================================================================
class C:
    """ANSI escape codes for terminal styling."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    UNDER   = "\033[4m"

    # Standard colors
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    # Bright colors
    BRED    = "\033[91m"
    BGREEN  = "\033[92m"
    BYELLOW = "\033[93m"
    BBLUE   = "\033[94m"
    BMAGENTA= "\033[95m"
    BCYAN   = "\033[96m"
    BWHITE  = "\033[97m"

    # Backgrounds
    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_BLUE   = "\033[44m"
    BG_CYAN   = "\033[46m"
    BG_MAGENTA= "\033[45m"
    BG_WHITE  = "\033[47m"
    BG_DARK   = "\033[40m"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print the Omnicide ASCII art banner."""
    banner = f"""
{C.BRED}{C.BOLD}
  ██████╗ ███╗   ███╗███╗   ██╗██╗ ██████╗██╗██████╗ ███████╗
 ██╔═══██╗████╗ ████║████╗  ██║██║██╔════╝██║██╔══██╗██╔════╝
 ██║   ██║██╔████╔██║██╔██╗ ██║██║██║     ██║██║  ██║█████╗
 ██║   ██║██║╚██╔╝██║██║╚██╗██║██║██║     ██║██║  ██║██╔══╝
 ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║╚██████╗██║██████╔╝███████╗
  ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝╚═╝╚═════╝ ╚══════╝
{C.RESET}
{C.DIM}  ───────────────────────────────────────────────────────────────{C.RESET}
{C.CYAN}  A Transformer Built From Absolute Zero{C.RESET}
{C.DIM}  Pure Python · Pure NumPy · No PyTorch · No TensorFlow{C.RESET}
{C.DIM}  ───────────────────────────────────────────────────────────────{C.RESET}
"""
    print(banner)


def print_divider(char="─", width=65, color=C.DIM):
    """Print a styled horizontal divider."""
    print(f"  {color}{char * width}{C.RESET}")


def print_header(text, color=C.BCYAN):
    """Print a section header."""
    print()
    print_divider()
    print(f"  {color}{C.BOLD}{text}{C.RESET}")
    print_divider()
    print()


def print_menu_item(number, icon, title, description, color=C.BWHITE):
    """Print a single styled menu option."""
    print(f"  {C.BOLD}{color}[{number}]{C.RESET}  {icon}  {C.BOLD}{title}{C.RESET}")
    print(f"        {C.DIM}{description}{C.RESET}")
    print()


def print_success(msg):
    """Print a success message."""
    print(f"\n  {C.BGREEN}✓{C.RESET} {C.GREEN}{msg}{C.RESET}")


def print_error(msg):
    """Print an error message."""
    print(f"\n  {C.BRED}✗{C.RESET} {C.RED}{msg}{C.RESET}")


def print_info(msg):
    """Print an info message."""
    print(f"  {C.BCYAN}→{C.RESET} {msg}")


def print_step(step_num, total, msg):
    """Print a numbered step with progress."""
    bar_width = 30
    filled = int(bar_width * step_num / total)
    bar = f"{C.BGREEN}{'█' * filled}{C.DIM}{'░' * (bar_width - filled)}{C.RESET}"
    print(f"\n  {C.BYELLOW}Step {step_num}/{total}{C.RESET} {bar}  {msg}")


def wait_for_enter():
    """Pause and wait for user to press Enter."""
    print(f"\n  {C.DIM}Press Enter to return to the main menu...{C.RESET}", end="")
    input()


def get_input(prompt, default=None):
    """Get user input with an optional default value."""
    if default is not None:
        user_in = input(f"  {C.BCYAN}?{C.RESET} {prompt} {C.DIM}(default: {default}){C.RESET}: ").strip()
        return user_in if user_in else str(default)
    else:
        return input(f"  {C.BCYAN}?{C.RESET} {prompt}: ").strip()


def animate_text(text, delay=0.015):
    """Print text with a typewriter animation effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# ==============================================================================
# MENU OPTIONS
# ==============================================================================

def option_forward_pass():
    """Run forward pass tests on both Transformer architectures."""
    clear_screen()
    print_header("🧪 FORWARD PASS TEST")

    from config import TransformerConfig
    from model import Transformer, DecoderOnlyTransformer

    # --- Encoder-Decoder ---
    print_step(1, 2, "Encoder-Decoder Transformer")
    print()

    config = TransformerConfig(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=32,
        num_heads=2,
        num_layers=2,
    )

    model = Transformer(config)
    np.random.seed(42)
    src_data = np.random.randint(0, config.src_vocab_size, (2, 5))
    tgt_data = np.random.randint(0, config.tgt_vocab_size, (2, 5))

    print_info("Running forward pass...")
    t0 = time.time()
    output = model.forward(src=src_data, tgt=tgt_data)
    elapsed = time.time() - t0

    total_params = sum(p.data.size for p in model.parameters())
    print()
    print(f"    {C.DIM}├─{C.RESET} Source shape:  {C.BYELLOW}{src_data.shape}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} Target shape:  {C.BYELLOW}{tgt_data.shape}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} Output shape:  {C.BGREEN}{output.shape}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} Parameters:    {C.BCYAN}{total_params:,}{C.RESET}")
    print(f"    {C.DIM}╰─{C.RESET} Time:          {C.BMAGENTA}{elapsed:.4f}s{C.RESET}")

    # --- Decoder-Only ---
    print_step(2, 2, "Decoder-Only Transformer (GPT-style)")
    print()

    config2 = TransformerConfig(
        vocab_size=50,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=128,
        max_seq_length=64,
    )

    model2 = DecoderOnlyTransformer(config2)
    np.random.seed(42)
    input_ids = np.random.randint(0, config2.vocab_size, (2, 10))

    print_info("Running forward pass...")
    t0 = time.time()
    logits = model2.forward(input_ids)
    elapsed = time.time() - t0

    total_params2 = sum(p.data.size for p in model2.parameters())
    print()
    print(f"    {C.DIM}├─{C.RESET} Input shape:   {C.BYELLOW}{input_ids.shape}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} Logits shape:  {C.BGREEN}{logits.shape}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} Parameters:    {C.BCYAN}{total_params2:,}{C.RESET}")
    print(f"    {C.DIM}╰─{C.RESET} Time:          {C.BMAGENTA}{elapsed:.4f}s{C.RESET}")

    print_success("Both architectures passed forward pass successfully!")
    wait_for_enter()


def option_train():
    """Run the full pretraining pipeline with configurable hyperparameters."""
    clear_screen()
    print_header("🏋️ PRETRAINING PIPELINE")

    print(f"  {C.BOLD}Configure your training run:{C.RESET}")
    print()

    # Get user configuration
    d_model = int(get_input("Embedding dimension (d_model)", 64))
    num_heads = int(get_input("Number of attention heads", 4))
    num_layers = int(get_input("Number of transformer blocks", 2))
    d_ff = int(get_input("Feed-forward hidden dim (d_ff)", 256))
    seq_length = int(get_input("Sequence length", 32))
    batch_size = int(get_input("Batch size", 4))
    num_steps = int(get_input("Training steps", 500))
    learning_rate = float(get_input("Learning rate", 1e-3))
    print_every = int(get_input("Print every N steps", 50))

    print()
    print_divider("═", color=C.BYELLOW)
    print(f"  {C.BYELLOW}{C.BOLD}  TRAINING CONFIGURATION{C.RESET}")
    print_divider("═", color=C.BYELLOW)
    print(f"    {C.DIM}├─{C.RESET} d_model:     {C.BCYAN}{d_model}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} num_heads:   {C.BCYAN}{num_heads}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} num_layers:  {C.BCYAN}{num_layers}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} d_ff:        {C.BCYAN}{d_ff}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} seq_length:  {C.BCYAN}{seq_length}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} batch_size:  {C.BCYAN}{batch_size}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} steps:       {C.BCYAN}{num_steps}{C.RESET}")
    print(f"    {C.DIM}╰─{C.RESET} lr:          {C.BCYAN}{learning_rate}{C.RESET}")
    print()

    confirm = get_input("Start training? (y/n)", "y")
    if confirm.lower() != 'y':
        print_info("Training cancelled.")
        wait_for_enter()
        return

    # --- Import and run training ---
    from config import TransformerConfig
    from model import DecoderOnlyTransformer
    from loss import CrossEntropyLoss
    from optim import Adam, clip_grad_norm
    from data import CharTokenizer, DataLoader
    from train import TRAINING_TEXT

    np.random.seed(42)

    # Step 1: Tokenization
    print_step(1, 5, "Tokenizing training data...")
    tokenizer = CharTokenizer()
    tokenizer.fit(TRAINING_TEXT)
    encoded_text = tokenizer.encode(TRAINING_TEXT)

    # Step 2: Model init
    print_step(2, 5, "Initializing model...")
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=seq_length * 2,
    )
    model = DecoderOnlyTransformer(config)
    total_params = sum(p.data.size for p in model.parameters())
    print_info(f"Total parameters: {C.BOLD}{total_params:,}{C.RESET}")

    # Step 3: Loss & Optimizer
    print_step(3, 5, "Setting up loss & optimizer...")
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Step 4: Data loader
    print_step(4, 5, "Building data loader...")
    dataloader = DataLoader(encoded_text, batch_size=batch_size, seq_length=seq_length)

    # Step 5: Training loop
    print_step(5, 5, "Starting training loop...")
    print()
    print_divider("═", color=C.BRED)
    print(f"  {C.BRED}{C.BOLD}  TRAINING{C.RESET}")
    print_divider("═", color=C.BRED)
    print()

    losses = []
    start_time = time.time()
    max_grad_norm = 1.0

    for step in range(num_steps):
        step_start = time.time()

        X, Y = dataloader.get_batch()
        logits = model.forward(X)
        loss = criterion(logits, Y)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), max_grad_norm)
        optimizer.step()

        loss_val = float(loss.data)
        losses.append(loss_val)
        step_time = time.time() - step_start

        if step % print_every == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            avg_loss = np.mean(losses[-print_every:]) if len(losses) >= print_every else np.mean(losses)

            # Color the loss based on progress
            if loss_val > 3.0:
                loss_color = C.BRED
            elif loss_val > 2.0:
                loss_color = C.BYELLOW
            else:
                loss_color = C.BGREEN

            # Progress bar
            progress = (step + 1) / num_steps
            bar_w = 20
            filled = int(bar_w * progress)
            bar = f"{C.BGREEN}{'█' * filled}{C.DIM}{'░' * (bar_w - filled)}{C.RESET}"

            print(
                f"  {bar} {C.DIM}Step{C.RESET} {C.BOLD}{step:4d}{C.RESET}/{num_steps} "
                f"{C.DIM}│{C.RESET} Loss: {loss_color}{loss_val:.4f}{C.RESET} "
                f"{C.DIM}│{C.RESET} Avg: {C.CYAN}{avg_loss:.4f}{C.RESET} "
                f"{C.DIM}│{C.RESET} ∇: {C.MAGENTA}{grad_norm:.3f}{C.RESET} "
                f"{C.DIM}│{C.RESET} {C.DIM}{step_time:.2f}s{C.RESET}"
            )

    # --- Results ---
    total_time = time.time() - start_time
    print()
    print_divider("═", color=C.BGREEN)
    print(f"  {C.BGREEN}{C.BOLD}  TRAINING COMPLETE{C.RESET}")
    print_divider("═", color=C.BGREEN)
    print()
    print(f"    {C.DIM}├─{C.RESET} Final loss:       {C.BGREEN}{losses[-1]:.4f}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} Initial loss:     {C.BRED}{losses[0]:.4f}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} Loss decrease:    {C.BCYAN}{losses[0] - losses[-1]:.4f}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} Total time:       {C.BMAGENTA}{total_time:.1f}s{C.RESET}")
    print(f"    {C.DIM}╰─{C.RESET} Expected random:  {C.DIM}~{np.log(tokenizer.vocab_size):.2f}{C.RESET}")

    # --- Text Generation ---
    print()
    print_header("✨ TEXT GENERATION")

    prompt = get_input("Enter a prompt", "First Citizen:")
    max_tokens = int(get_input("Max new tokens to generate", 200))
    temperature = float(get_input("Temperature (0.1-2.0)", 0.8))

    prompt_ids = np.array([tokenizer.encode(prompt)])
    print()
    print_info("Generating...")
    print()

    generated_ids = model.generate(prompt_ids, max_new_tokens=max_tokens, temperature=temperature)
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"  {C.DIM}┌{'─' * 63}┐{C.RESET}")
    # Word wrap the generated text
    for line in _wrap_text(generated_text, 61):
        print(f"  {C.DIM}│{C.RESET} {C.BWHITE}{line:<61}{C.RESET} {C.DIM}│{C.RESET}")
    print(f"  {C.DIM}└{'─' * 63}┘{C.RESET}")

    print_success(f"Generated {len(generated_text)} characters from prompt '{prompt}'")
    wait_for_enter()


def option_generate():
    """Generate text from a pretrained model (quick train + generate)."""
    clear_screen()
    print_header("✨ TEXT GENERATION (Quick)")

    print(f"  {C.DIM}This will quickly train a small model and then generate text.{C.RESET}")
    print(f"  {C.DIM}For higher quality, use the full training pipeline (option 2).{C.RESET}")
    print()

    prompt = get_input("Enter a generation prompt", "First Citizen:")
    max_tokens = int(get_input("Max new tokens", 200))
    temperature = float(get_input("Temperature (0.1-2.0)", 0.8))
    quick_steps = int(get_input("Quick training steps", 200))

    from config import TransformerConfig
    from model import DecoderOnlyTransformer
    from loss import CrossEntropyLoss
    from optim import Adam, clip_grad_norm
    from data import CharTokenizer, DataLoader
    from train import TRAINING_TEXT

    np.random.seed(42)

    print()
    print_info("Tokenizing...")
    tokenizer = CharTokenizer()
    tokenizer.fit(TRAINING_TEXT)
    encoded_text = tokenizer.encode(TRAINING_TEXT)

    print_info("Initializing model...")
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64, num_heads=4, num_layers=2,
        d_ff=256, max_seq_length=64,
    )
    model = DecoderOnlyTransformer(config)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    dataloader = DataLoader(encoded_text, batch_size=4, seq_length=32)

    print_info(f"Quick training ({quick_steps} steps)...")
    print()

    start_time = time.time()
    for step in range(quick_steps):
        X, Y = dataloader.get_batch()
        logits = model.forward(X)
        loss = criterion(logits, Y)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == quick_steps - 1:
            progress = (step + 1) / quick_steps
            bar_w = 30
            filled = int(bar_w * progress)
            bar = f"{C.BGREEN}{'█' * filled}{C.DIM}{'░' * (bar_w - filled)}{C.RESET}"
            print(f"  {bar} {C.DIM}step {step}/{quick_steps}{C.RESET} loss={C.BYELLOW}{float(loss.data):.4f}{C.RESET}")

    elapsed = time.time() - start_time
    print()
    print_success(f"Training complete in {elapsed:.1f}s (final loss: {float(loss.data):.4f})")

    # Generate
    print()
    print_info("Generating text...")
    print()

    prompt_ids = np.array([tokenizer.encode(prompt)])
    generated_ids = model.generate(prompt_ids, max_new_tokens=max_tokens, temperature=temperature)
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"  {C.DIM}Prompt: {C.RESET}{C.BOLD}{prompt}{C.RESET}")
    print()
    print(f"  {C.DIM}┌{'─' * 63}┐{C.RESET}")
    for line in _wrap_text(generated_text, 61):
        print(f"  {C.DIM}│{C.RESET} {C.BWHITE}{line:<61}{C.RESET} {C.DIM}│{C.RESET}")
    print(f"  {C.DIM}└{'─' * 63}┘{C.RESET}")

    print_success(f"Generated {len(generated_text)} characters!")
    wait_for_enter()


def option_autograd_demo():
    """Run the scalar autograd engine demo."""
    clear_screen()
    print_header("🔬 SCALAR AUTOGRAD ENGINE DEMO")

    from engine import Value

    print(f"  {C.BOLD}Demonstrating backpropagation on a simple computation graph:{C.RESET}")
    print()
    print(f"  {C.DIM}Equation:{C.RESET}  {C.BCYAN}out = (x * y) + relu(x){C.RESET}")
    print(f"  {C.DIM}Where:{C.RESET}     {C.BYELLOW}x = -2.0{C.RESET}, {C.BYELLOW}y = 3.0{C.RESET}")
    print()

    x = Value(-2.0, label='x')
    y = Value(3.0, label='y')

    # Forward pass
    print(f"  {C.BGREEN}▶ Forward Pass:{C.RESET}")
    xy = x * y; xy.label = 'x*y'
    relu_x = x.relu(); relu_x.label = 'relu(x)'
    out = xy + relu_x; out.label = 'out'

    print(f"    {C.DIM}├─{C.RESET} x * y     = {C.BYELLOW}{xy.data}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} relu(x)   = {C.BYELLOW}{relu_x.data}{C.RESET}")
    print(f"    {C.DIM}╰─{C.RESET} out       = {C.BGREEN}{out.data}{C.RESET}")

    print()

    # Backward pass
    print(f"  {C.BRED}◀ Backward Pass:{C.RESET}")
    out.backward()

    print(f"    {C.DIM}├─{C.RESET} ∂out/∂x   = {C.BCYAN}{x.grad}{C.RESET}  {C.DIM}(∂(x*y)/∂x + ∂relu(x)/∂x = y + 0 = 3.0){C.RESET}")
    print(f"    {C.DIM}╰─{C.RESET} ∂out/∂y   = {C.BCYAN}{y.grad}{C.RESET}  {C.DIM}(∂(x*y)/∂y = x = -2.0){C.RESET}")

    print()
    print_divider()

    # Computation graph visualization
    print()
    print(f"  {C.BOLD}Computation Graph:{C.RESET}")
    print()
    print(f"    {C.BYELLOW}x = -2.0{C.RESET} ──┐")
    print(f"    {C.DIM}(grad: {x.grad}){C.RESET}  ├──▶ {C.BCYAN}[  ×  ]{C.RESET} ──▶ {C.DIM}x*y = {xy.data}{C.RESET} ──┐")
    print(f"    {C.BYELLOW}y =  3.0{C.RESET} ──┘                                 │")
    print(f"    {C.DIM}(grad: {y.grad}){C.RESET}                                    ├──▶ {C.BGREEN}[  +  ]{C.RESET} ──▶ {C.BOLD}out = {out.data}{C.RESET}")
    print(f"                                                │")
    print(f"    {C.BYELLOW}x = -2.0{C.RESET} ────▶ {C.BCYAN}[ReLU]{C.RESET} ──▶ {C.DIM}relu = {relu_x.data}{C.RESET}  ──┘")

    # --- Interactive playground ---
    print()
    print_divider()
    print()
    print(f"  {C.BOLD}Interactive Playground:{C.RESET}")
    print(f"  {C.DIM}Try your own values! (press Enter to skip){C.RESET}")
    print()

    custom_x = get_input("Enter x value", None)
    if custom_x:
        custom_y = get_input("Enter y value", "3.0")

        x2 = Value(float(custom_x), label='x')
        y2 = Value(float(custom_y), label='y')
        xy2 = x2 * y2
        relu_x2 = x2.relu()
        out2 = xy2 + relu_x2
        out2.backward()

        print()
        print(f"    {C.BOLD}Results:{C.RESET}")
        print(f"    {C.DIM}├─{C.RESET} out       = {C.BGREEN}{out2.data}{C.RESET}")
        print(f"    {C.DIM}├─{C.RESET} ∂out/∂x   = {C.BCYAN}{x2.grad}{C.RESET}")
        print(f"    {C.DIM}╰─{C.RESET} ∂out/∂y   = {C.BCYAN}{y2.grad}{C.RESET}")

    wait_for_enter()


def option_inspect_model():
    """Inspect model architecture and parameter counts."""
    clear_screen()
    print_header("🔍 MODEL INSPECTOR")

    from config import TransformerConfig
    from model import Transformer, DecoderOnlyTransformer

    print(f"  {C.BOLD}Which model do you want to inspect?{C.RESET}")
    print()
    print(f"    {C.BOLD}[1]{C.RESET}  Encoder-Decoder Transformer  {C.DIM}(Vaswani et al., 2017){C.RESET}")
    print(f"    {C.BOLD}[2]{C.RESET}  Decoder-Only Transformer     {C.DIM}(GPT-style){C.RESET}")
    print()

    choice = get_input("Select model", "2")
    print()

    d_model = int(get_input("d_model", 64))
    num_heads = int(get_input("num_heads", 4))
    num_layers = int(get_input("num_layers", 2))
    d_ff = int(get_input("d_ff", 256))
    vocab_size = int(get_input("vocab_size", 100))

    print()

    if choice == "1":
        config = TransformerConfig(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
        )
        model = Transformer(config)
        model_name = "Encoder-Decoder Transformer"
    else:
        config = TransformerConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=64,
        )
        model = DecoderOnlyTransformer(config)
        model_name = "Decoder-Only Transformer"

    params = model.parameters()
    total_params = sum(p.data.size for p in params)
    trainable = sum(p.data.size for p in params if p.requires_grad)

    print_divider("═", color=C.BCYAN)
    print(f"  {C.BCYAN}{C.BOLD}  {model_name}{C.RESET}")
    print_divider("═", color=C.BCYAN)
    print()
    print(f"    {C.BOLD}Architecture:{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} d_model:        {C.BYELLOW}{d_model}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} num_heads:      {C.BYELLOW}{num_heads}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} head_dim:       {C.BYELLOW}{d_model // num_heads}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} num_layers:     {C.BYELLOW}{num_layers}{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} d_ff:           {C.BYELLOW}{d_ff}{C.RESET}")
    print(f"    {C.DIM}╰─{C.RESET} vocab_size:     {C.BYELLOW}{vocab_size}{C.RESET}")
    print()
    print(f"    {C.BOLD}Parameters:{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} Total:          {C.BGREEN}{total_params:,}{C.RESET}")
    print(f"    {C.DIM}╰─{C.RESET} Trainable:      {C.BGREEN}{trainable:,}{C.RESET}")

    # Per-parameter breakdown
    print()
    print(f"    {C.BOLD}Parameter Breakdown:{C.RESET}")

    param_groups = {}
    for p in params:
        shape_str = str(p.data.shape)
        size = p.data.size
        name = getattr(p, '_name', shape_str)
        if shape_str not in param_groups:
            param_groups[shape_str] = {'count': 0, 'total_size': 0}
        param_groups[shape_str]['count'] += 1
        param_groups[shape_str]['total_size'] += size

    print(f"    {'Shape':<25} {'Count':<8} {'Size':<15}")
    print(f"    {C.DIM}{'─' * 48}{C.RESET}")
    for shape, info in sorted(param_groups.items(), key=lambda x: -x[1]['total_size']):
        pct = info['total_size'] / total_params * 100
        bar_w = 15
        filled = int(bar_w * pct / 100)
        bar = f"{C.BGREEN}{'█' * filled}{C.DIM}{'░' * (bar_w - filled)}{C.RESET}"
        print(f"    {C.CYAN}{shape:<25}{C.RESET} {info['count']:<8} {info['total_size']:<10,} {bar} {pct:.1f}%")

    # GPT-2 comparison
    print()
    print(f"    {C.BOLD}Comparison with GPT-2:{C.RESET}")
    gpt2_params = 117_000_000
    ratio = gpt2_params / total_params
    print(f"    {C.DIM}├─{C.RESET} Your model:     {C.BCYAN}{total_params:>15,} params{C.RESET}")
    print(f"    {C.DIM}├─{C.RESET} GPT-2 (small):  {C.BYELLOW}{gpt2_params:>15,} params{C.RESET}")
    print(f"    {C.DIM}╰─{C.RESET} Scale factor:   {C.BRED}{ratio:>14,.0f}×{C.RESET}")

    wait_for_enter()


def option_tensor_autograd():
    """Demonstrate the Tensor autograd engine with matrix operations."""
    clear_screen()
    print_header("🧬 TENSOR AUTOGRAD DEMO")

    from tensor import Tensor

    print(f"  {C.BOLD}Demonstrating N-dimensional autograd with Tensor class:{C.RESET}")
    print()

    # Demo 1: Matrix multiply + backward
    print(f"  {C.BYELLOW}Demo 1: Matrix Multiplication Gradients{C.RESET}")
    print(f"  {C.DIM}C = A @ B, then backprop to get ∂C/∂A and ∂C/∂B{C.RESET}")
    print()

    np.random.seed(42)
    A = Tensor(np.random.randn(2, 3), requires_grad=True)
    B = Tensor(np.random.randn(3, 2), requires_grad=True)

    C = A @ B                    # matmul
    loss = C.sum()               # reduce to scalar
    loss.backward()

    print(f"    A shape: {C.BCYAN}{A.data.shape}{C.RESET}  B shape: {C.BCYAN}{B.data.shape}{C.RESET}")
    print(f"    C = A @ B shape: {C.BGREEN}{C.data.shape}{C.RESET}")
    print(f"    loss = sum(C) = {C.BYELLOW}{loss.data:.4f}{C.RESET}")
    print()
    print(f"    ∂loss/∂A (shape {A.grad.shape}):")
    for row in A.grad:
        formatted = "  ".join(f"{v:>7.4f}" for v in row)
        print(f"      [{formatted}]")
    print()
    print(f"    ∂loss/∂B (shape {B.grad.shape}):")
    for row in B.grad:
        formatted = "  ".join(f"{v:>7.4f}" for v in row)
        print(f"      [{formatted}]")

    # Demo 2: Chain of operations
    print()
    print_divider()
    print()
    print(f"  {C.BYELLOW}Demo 2: Chained Operations{C.RESET}")
    print(f"  {C.DIM}y = relu(x @ W + b), loss = sum(y){C.RESET}")
    print()

    x = Tensor(np.random.randn(1, 4), requires_grad=False)
    W = Tensor(np.random.randn(4, 3), requires_grad=True)
    b = Tensor(np.zeros((1, 3)), requires_grad=True)

    from tensor import tensor_relu

    z = (x @ W) + b
    y = tensor_relu(z)
    loss = y.sum()
    loss.backward()

    print(f"    x: {x.data.flatten()}")
    print(f"    z = x @ W + b: {z.data.flatten()}")
    print(f"    y = relu(z):   {y.data.flatten()}")
    print(f"    loss = {C.BGREEN}{loss.data:.4f}{C.RESET}")
    print()
    print(f"    ∂loss/∂W (shape {W.grad.shape}):")
    for row in W.grad:
        formatted = "  ".join(f"{v:>7.4f}" for v in row)
        print(f"      [{formatted}]")
    print(f"    ∂loss/∂b: {b.grad.flatten()}")

    print()
    print_success("Tensor autograd working correctly — gradients flow through matmul, add, relu!")
    wait_for_enter()


def option_architecture():
    """Display the Transformer architecture diagram."""
    clear_screen()
    print_header("📐 ARCHITECTURE DIAGRAM")

    arch = f"""
  {C.BOLD}Decoder-Only Transformer (GPT-style){C.RESET}
  {C.DIM}The architecture used for next-token prediction{C.RESET}

  {C.BYELLOW}Input Text{C.RESET}
      │
      ▼
  {C.BOLD}┌───────────────────────────────┐{C.RESET}
  {C.BOLD}│{C.RESET}  {C.BCYAN}Character Tokenizer{C.RESET}          {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}  {C.DIM}"hello" → [7, 4, 11, 11, 14]{C.RESET} {C.BOLD}│{C.RESET}
  {C.BOLD}└───────────────┬───────────────┘{C.RESET}
                  │
                  ▼
  {C.BOLD}┌───────────────────────────────┐{C.RESET}
  {C.BOLD}│{C.RESET}  {C.BGREEN}Token Embedding{C.RESET}              {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}  {C.DIM}ID → dense vector (d_model){C.RESET}   {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}                               {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}  {C.BGREEN}+ Positional Encoding{C.RESET}        {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}  {C.DIM}sin/cos position signals{C.RESET}      {C.BOLD}│{C.RESET}
  {C.BOLD}└───────────────┬───────────────┘{C.RESET}
                  │
         ╔════════╧════════╗
         ║  {C.BRED}× N layers{C.RESET}      ║
         ╠═════════════════╣
         ║                 ║
         ║  {C.BCYAN}Layer Norm{C.RESET}      ║
         ║       │         ║
         ║  {C.BYELLOW}Multi-Head{C.RESET}     ║
         ║  {C.BYELLOW}Self-Attention{C.RESET}  ║
         ║  {C.DIM}(causal mask){C.RESET}   ║
         ║       │         ║
         ║  {C.DIM}+ residual{C.RESET}      ║
         ║       │         ║
         ║  {C.BCYAN}Layer Norm{C.RESET}      ║
         ║       │         ║
         ║  {C.BMAGENTA}Feed-Forward{C.RESET}   ║
         ║  {C.DIM}(expand→ReLU{C.RESET}    ║
         ║  {C.DIM} →compress){C.RESET}     ║
         ║       │         ║
         ║  {C.DIM}+ residual{C.RESET}      ║
         ║                 ║
         ╚════════╤════════╝
                  │
                  ▼
  {C.BOLD}┌───────────────────────────────┐{C.RESET}
  {C.BOLD}│{C.RESET}  {C.BCYAN}Final Layer Norm{C.RESET}             {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}  {C.BGREEN}Linear → Logits{C.RESET}             {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}  {C.DIM}(d_model → vocab_size){C.RESET}       {C.BOLD}│{C.RESET}
  {C.BOLD}└───────────────┬───────────────┘{C.RESET}
                  │
                  ▼
  {C.BOLD}┌───────────────────────────────┐{C.RESET}
  {C.BOLD}│{C.RESET}  {C.BRED}Cross-Entropy Loss{C.RESET}           {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}  {C.DIM}softmax + NLL{C.RESET}                {C.BOLD}│{C.RESET}
  {C.BOLD}└───────────────┬───────────────┘{C.RESET}
                  │
                  ▼
  {C.BOLD}┌───────────────────────────────┐{C.RESET}
  {C.BOLD}│{C.RESET}  {C.BYELLOW}loss.backward(){C.RESET}              {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}  {C.DIM}autograd through everything{C.RESET}   {C.BOLD}│{C.RESET}
  {C.BOLD}└───────────────┬───────────────┘{C.RESET}
                  │
                  ▼
  {C.BOLD}┌───────────────────────────────┐{C.RESET}
  {C.BOLD}│{C.RESET}  {C.BMAGENTA}Adam Optimizer{C.RESET}               {C.BOLD}│{C.RESET}
  {C.BOLD}│{C.RESET}  {C.DIM}θ -= lr · m̂/(√v̂ + ε){C.RESET}        {C.BOLD}│{C.RESET}
  {C.BOLD}└───────────────────────────────┘{C.RESET}

  {C.BOLD}Attention Formula:{C.RESET}
  {C.BCYAN}Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V{C.RESET}

  {C.BOLD}Cross-Entropy Gradient (The Beautiful One):{C.RESET}
  {C.BRED}∂L/∂z = softmax(z) − one_hot(target){C.RESET}
"""
    print(arch)
    wait_for_enter()


def option_math():
    """Show the mathematical equations implemented in Omnicide."""
    clear_screen()
    print_header("🧮 THE MATH UNDER THE HOOD")

    equations = f"""
  {C.BYELLOW}{C.BOLD}1. Scaled Dot-Product Attention{C.RESET}
  {C.DIM}   The core operation of every Transformer:{C.RESET}

     {C.BCYAN}Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V{C.RESET}

     {C.DIM}Q = queries, K = keys, V = values
     d_k = dimension of keys (prevents dot products from getting too large){C.RESET}

  {C.DIM}{'─' * 60}{C.RESET}

  {C.BYELLOW}{C.BOLD}2. Backprop Through MatMul{C.RESET}
  {C.DIM}   How gradients flow backwards through matrix multiplication:{C.RESET}

     {C.BCYAN}If C = A @ B:{C.RESET}
     {C.BGREEN}∂L/∂A = ∂L/∂C @ Bᵀ{C.RESET}
     {C.BGREEN}∂L/∂B = Aᵀ @ ∂L/∂C{C.RESET}

  {C.DIM}{'─' * 60}{C.RESET}

  {C.BYELLOW}{C.BOLD}3. Softmax Backward{C.RESET}
  {C.DIM}   The Jacobian-vector product for softmax:{C.RESET}

     {C.BCYAN}s = softmax(x){C.RESET}
     {C.BGREEN}∂L/∂x = s ⊙ (∂L/∂s − ⟨∂L/∂s, s⟩){C.RESET}

  {C.DIM}{'─' * 60}{C.RESET}

  {C.BYELLOW}{C.BOLD}4. Layer Normalization{C.RESET}
  {C.DIM}   Normalizes activations, the backward pass is complex:{C.RESET}

     {C.BCYAN}forward: x̂ = (x − μ) / σ,  y = γ · x̂ + β{C.RESET}
     {C.BGREEN}backward: ∂L/∂x = (1/Nσ) · [N·dx̂ − Σ(dx̂) − x̂·Σ(dx̂·x̂)]{C.RESET}

  {C.DIM}{'─' * 60}{C.RESET}

  {C.BYELLOW}{C.BOLD}5. Cross-Entropy Loss (The Beautiful One){C.RESET}
  {C.DIM}   The gradient is elegantly simple despite the complex forward:{C.RESET}

     {C.BCYAN}L = −Σ log(softmax(z)[target]){C.RESET}
     {C.BRED}∂L/∂z = softmax(z) − one_hot(target){C.RESET}

     {C.DIM}"The model's probability distribution minus the truth."
     This single equation drives all learning in GPT-style models.{C.RESET}

  {C.DIM}{'─' * 60}{C.RESET}

  {C.BYELLOW}{C.BOLD}6. Adam Optimizer{C.RESET}
  {C.DIM}   Adaptive moment estimation (Kingma & Ba, 2014):{C.RESET}

     {C.BCYAN}m_t = β₁·m_(t-1) + (1−β₁)·g_t{C.RESET}          {C.DIM}(momentum){C.RESET}
     {C.BCYAN}v_t = β₂·v_(t-1) + (1−β₂)·g_t²{C.RESET}         {C.DIM}(adaptive lr){C.RESET}
     {C.BGREEN}θ_t = θ_(t-1) − lr · m̂_t / (√v̂_t + ε){C.RESET}  {C.DIM}(update){C.RESET}

  {C.DIM}{'─' * 60}{C.RESET}

  {C.BYELLOW}{C.BOLD}7. Gradient Clipping{C.RESET}
  {C.DIM}   Prevents exploding gradients:{C.RESET}

     {C.BCYAN}global_norm = √(Σ ‖∇θᵢ‖²){C.RESET}
     {C.BGREEN}if global_norm > max_norm: ∇θᵢ *= max_norm / global_norm{C.RESET}
"""
    print(equations)
    wait_for_enter()


# ==============================================================================
# CALCULATOR SUITE
# ==============================================================================

def _fmt_num(n, decimals=1):
    """Format a large number with human-readable suffix."""
    if n < 1e3:
        return f"{n:,.{decimals}f}"
    elif n < 1e6:
        return f"{n / 1e3:,.{decimals}f}K"
    elif n < 1e9:
        return f"{n / 1e6:,.{decimals}f}M"
    elif n < 1e12:
        return f"{n / 1e9:,.{decimals}f}B"
    elif n < 1e15:
        return f"{n / 1e12:,.{decimals}f}T"
    elif n < 1e18:
        return f"{n / 1e15:,.{decimals}f}P"
    else:
        return f"{n / 1e18:,.{decimals}f}E"


def _fmt_bytes(b):
    """Format bytes into human-readable form."""
    if b < 1024:
        return f"{b:.0f} B"
    elif b < 1024**2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    elif b < 1024**4:
        return f"{b / 1024**3:.2f} GB"
    else:
        return f"{b / 1024**4:.2f} TB"


def _fmt_time(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    elif seconds < 86400 * 365:
        return f"{seconds / 86400:.1f} days"
    else:
        return f"{seconds / (86400 * 365):.2f} years"


def _count_transformer_params(vocab_size, d_model, num_layers, d_ff, num_heads):
    """Count parameters in a decoder-only Transformer analytically."""
    # Embedding: vocab_size * d_model
    emb = vocab_size * d_model

    # Per-layer:
    #   Multi-Head Attention: 4 projections (Q, K, V, O) each d_model x d_model + biases
    attn_per_layer = 4 * (d_model * d_model + d_model)
    #   FFN: two linear layers (d_model -> d_ff -> d_model) + biases
    ffn_per_layer = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    #   Two LayerNorms: 2 * (gamma + beta) = 2 * 2 * d_model
    ln_per_layer = 2 * (d_model + d_model)

    per_layer = attn_per_layer + ffn_per_layer + ln_per_layer
    all_layers = num_layers * per_layer

    # Final layer norm + output projection
    final_ln = d_model + d_model
    output_proj = d_model * vocab_size + vocab_size

    total = emb + all_layers + final_ln + output_proj
    return {
        'total': total,
        'embedding': emb,
        'per_layer': per_layer,
        'attn_per_layer': attn_per_layer,
        'ffn_per_layer': ffn_per_layer,
        'ln_per_layer': ln_per_layer,
        'all_layers': all_layers,
        'final_ln': final_ln,
        'output_proj': output_proj,
    }


def _get_model_specs(title="Model Specifications"):
    """Prompt user for detailed model specifications. Returns a dict."""
    print(f"  {C.BOLD}{title}:{C.RESET}")
    print()
    vocab_size = int(get_input("Vocabulary size", 50257))
    d_model = int(get_input("Embedding dimension (d_model)", 768))
    num_heads = int(get_input("Number of attention heads", 12))
    num_layers = int(get_input("Number of transformer layers (depth)", 12))
    d_ff = int(get_input("Feed-forward hidden dim (d_ff, typically 4*d_model)", d_model * 4))
    max_seq_length = int(get_input("Max sequence length (context window)", 1024))
    return {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_ff': d_ff,
        'max_seq_length': max_seq_length,
        'head_dim': d_model // num_heads,
    }


def _print_model_card(specs, params):
    """Print a formatted model specification card."""
    print()
    print_divider("=", color=C.BCYAN)
    print(f"  {C.BCYAN}{C.BOLD}  MODEL CARD{C.RESET}")
    print_divider("=", color=C.BCYAN)
    print(f"    {C.DIM}|--{C.RESET} vocab_size:      {C.BYELLOW}{specs['vocab_size']:,}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} d_model:         {C.BYELLOW}{specs['d_model']}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} num_heads:       {C.BYELLOW}{specs['num_heads']}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} head_dim:        {C.BYELLOW}{specs['head_dim']}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} num_layers:      {C.BYELLOW}{specs['num_layers']}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} d_ff:            {C.BYELLOW}{specs['d_ff']}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} max_seq_length:  {C.BYELLOW}{specs['max_seq_length']}{C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} Total params:    {C.BGREEN}{C.BOLD}{params['total']:,}{C.RESET}  ({_fmt_num(params['total'])})")
    print()


# --- Hardware Presets ---
HARDWARE_PRESETS = {
    '1': {'name': 'NVIDIA RTX 4090',            'tflops_fp32': 82.6,  'tflops_fp16': 165.2, 'vram_gb': 24,   'mem_bw_gbs': 1008},
    '2': {'name': 'NVIDIA A100 (80GB)',          'tflops_fp32': 19.5,  'tflops_fp16': 312.0, 'vram_gb': 80,   'mem_bw_gbs': 2039},
    '3': {'name': 'NVIDIA H100 (SXM)',           'tflops_fp32': 67.0,  'tflops_fp16': 989.0, 'vram_gb': 80,   'mem_bw_gbs': 3350},
    '4': {'name': 'NVIDIA RTX 3090',             'tflops_fp32': 35.6,  'tflops_fp16': 71.0,  'vram_gb': 24,   'mem_bw_gbs': 936},
    '5': {'name': 'NVIDIA V100 (32GB)',          'tflops_fp32': 15.7,  'tflops_fp16': 125.0, 'vram_gb': 32,   'mem_bw_gbs': 900},
    '6': {'name': 'Apple M2 Ultra (GPU)',        'tflops_fp32': 27.2,  'tflops_fp16': 54.4,  'vram_gb': 192,  'mem_bw_gbs': 800},
    '7': {'name': 'Google TPU v4',               'tflops_fp32': 123.0, 'tflops_fp16': 275.0, 'vram_gb': 32,   'mem_bw_gbs': 1200},
    '8': {'name': 'CPU Only (rough estimate)',   'tflops_fp32': 0.5,   'tflops_fp16': 0.5,   'vram_gb': 64,   'mem_bw_gbs': 50},
}


def _select_hardware():
    """Prompt user to select a hardware preset or enter custom specs."""
    print(f"  {C.BOLD}Select hardware:{C.RESET}")
    print()
    for key, hw in HARDWARE_PRESETS.items():
        print(f"    {C.BOLD}[{key}]{C.RESET}  {hw['name']:<30} {C.DIM}({hw['tflops_fp16']:.0f} TFLOPS fp16, {hw['vram_gb']}GB VRAM){C.RESET}")
    print(f"    {C.BOLD}[9]{C.RESET}  {C.BYELLOW}Custom hardware specs{C.RESET}")
    print()
    ch = get_input("Hardware choice", "1")

    if ch in HARDWARE_PRESETS:
        hw = HARDWARE_PRESETS[ch]
        print_info(f"Selected: {C.BOLD}{hw['name']}{C.RESET}")
        return hw
    else:
        name = get_input("Hardware name", "Custom GPU")
        tflops = float(get_input("Peak TFLOPS (fp16/bf16)", 100))
        vram = float(get_input("VRAM / Memory (GB)", 24))
        mem_bw = float(get_input("Memory bandwidth (GB/s)", 900))
        return {'name': name, 'tflops_fp32': tflops / 2, 'tflops_fp16': tflops, 'vram_gb': vram, 'mem_bw_gbs': mem_bw}


def calc_training_time():
    """Training Time Calculator -- given model specs and training config."""
    clear_screen()
    print_header("TRAINING TIME CALCULATOR")

    print(f"  {C.DIM}Estimates wall-clock training time based on model architecture,{C.RESET}")
    print(f"  {C.DIM}training config, and hardware. Uses the ~6ND approximation.{C.RESET}")
    print()

    specs = _get_model_specs()
    params = _count_transformer_params(
        specs['vocab_size'], specs['d_model'], specs['num_layers'],
        specs['d_ff'], specs['num_heads']
    )
    _print_model_card(specs, params)

    # Training config
    print(f"  {C.BOLD}Training Configuration:{C.RESET}")
    print()
    batch_size = int(get_input("Batch size (per device)", 32))
    seq_length = int(get_input("Training sequence length", specs['max_seq_length']))
    num_epochs = float(get_input("Number of epochs (or fraction)", 1.0))
    dataset_tokens = int(float(get_input("Dataset size in tokens (e.g. 1e9 for 1B)", 1e9)))
    num_gpus = int(get_input("Number of GPUs/accelerators", 1))
    gpu_utilization = float(get_input("GPU utilization factor (0.0-1.0, typically 0.3-0.5)", 0.35))
    precision = get_input("Training precision (fp32 / fp16 / bf16)", "fp16")
    print()

    hw = _select_hardware()

    # --- Calculations ---
    N = params['total']
    D = int(dataset_tokens * num_epochs)

    # Approximation: total training FLOPs ~ 6 * N * D
    total_flops = 6 * N * D

    # More detailed breakdown
    flops_fwd_per_token = 2 * N
    flops_bwd_per_token = 4 * N
    flops_per_token = flops_fwd_per_token + flops_bwd_per_token
    tokens_per_step = batch_size * seq_length
    flops_per_step = flops_per_token * tokens_per_step
    total_steps = D // tokens_per_step

    # Hardware throughput
    if precision == "fp32":
        peak_tflops = hw['tflops_fp32']
    else:
        peak_tflops = hw['tflops_fp16']
    effective_tflops = peak_tflops * gpu_utilization * num_gpus
    effective_flops = effective_tflops * 1e12

    # Wall-clock time
    if effective_flops > 0:
        total_seconds = total_flops / effective_flops
    else:
        total_seconds = float('inf')

    seconds_per_step = total_seconds / max(total_steps, 1)
    tokens_per_second = D / max(total_seconds, 1e-9)

    # --- Display Results ---
    print()
    print_divider("=", color=C.BGREEN)
    print(f"  {C.BGREEN}{C.BOLD}  TRAINING TIME ESTIMATE{C.RESET}")
    print_divider("=", color=C.BGREEN)
    print()

    print(f"    {C.BOLD}Compute Breakdown:{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Parameters (N):       {C.BCYAN}{N:>20,}{C.RESET}  ({_fmt_num(N)})")
    print(f"    {C.DIM}|--{C.RESET} Total tokens (D):     {C.BCYAN}{D:>20,}{C.RESET}  ({_fmt_num(D)})")
    print(f"    {C.DIM}|--{C.RESET} FLOPs/token (6N):     {C.BCYAN}{flops_per_token:>20,}{C.RESET}  ({_fmt_num(flops_per_token)})")
    print(f"    {C.DIM}|--{C.RESET} Tokens/step:          {C.BCYAN}{tokens_per_step:>20,}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Total steps:          {C.BCYAN}{total_steps:>20,}{C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} Total FLOPs (6ND):    {C.BYELLOW}{C.BOLD}{total_flops:>20.3e}{C.RESET}  ({_fmt_num(total_flops)})")
    print()

    print(f"    {C.BOLD}Hardware:{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Device:               {C.BMAGENTA}{hw['name']}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} GPUs:                 {C.BMAGENTA}{num_gpus}{C.RESET}")
    prec_label = 'fp16' if precision != 'fp32' else 'fp32'
    print(f"    {C.DIM}|--{C.RESET} Peak TFLOPS ({prec_label}):   {C.BMAGENTA}{peak_tflops:.1f}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Utilization:          {C.BMAGENTA}{gpu_utilization * 100:.0f}%{C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} Effective TFLOPS:     {C.BMAGENTA}{effective_tflops:.1f}{C.RESET}")
    print()

    # Time estimate with color-coded severity
    if total_seconds < 3600:
        time_color = C.BGREEN
    elif total_seconds < 86400:
        time_color = C.BYELLOW
    elif total_seconds < 86400 * 30:
        time_color = C.BRED
    else:
        time_color = C.BMAGENTA

    print(f"    {C.BOLD}Estimated Training Time:{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Wall-clock:           {time_color}{C.BOLD}{_fmt_time(total_seconds)}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Seconds/step:         {C.BCYAN}{seconds_per_step:.4f}s{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Tokens/second:        {C.BCYAN}{tokens_per_second:,.0f}{C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} GPU-hours:            {C.BYELLOW}{(total_seconds / 3600) * num_gpus:,.1f}{C.RESET}")

    # Cost estimate
    print()
    print(f"    {C.BOLD}Cost Estimate (cloud):{C.RESET}")
    cost_per_gpu_hr = float(get_input("Cost per GPU-hour ($)", 2.50))
    total_cost = (total_seconds / 3600) * num_gpus * cost_per_gpu_hr
    print(f"    {C.DIM}`--{C.RESET} Estimated cost:       {C.BYELLOW}{C.BOLD}${total_cost:,.2f}{C.RESET}")

    wait_for_enter()


def calc_inference():
    """Inference Calculator -- latency, throughput, and memory."""
    clear_screen()
    print_header("INFERENCE CALCULATOR")

    print(f"  {C.DIM}Estimates inference latency, throughput, and KV cache memory.{C.RESET}")
    print(f"  {C.DIM}Covers both prefill (prompt processing) and decode (generation).{C.RESET}")
    print()

    specs = _get_model_specs()
    params = _count_transformer_params(
        specs['vocab_size'], specs['d_model'], specs['num_layers'],
        specs['d_ff'], specs['num_heads']
    )
    _print_model_card(specs, params)

    # Inference config
    print(f"  {C.BOLD}Inference Configuration:{C.RESET}")
    print()
    prompt_length = int(get_input("Prompt length (tokens)", 512))
    gen_tokens = int(get_input("Tokens to generate", 256))
    batch_size = int(get_input("Batch size", 1))
    precision = get_input("Precision (fp32 / fp16 / int8 / int4)", "fp16")
    print()

    hw = _select_hardware()

    N = params['total']
    L = specs['num_layers']
    d = specs['d_model']
    h = specs['num_heads']
    d_k = specs['head_dim']
    S = specs['max_seq_length']

    # Bytes per parameter based on precision
    bytes_per_param = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1, 'int4': 0.5}.get(precision, 2)

    # --- Model Memory ---
    model_bytes = N * bytes_per_param

    # --- KV Cache ---
    total_seq_len = prompt_length + gen_tokens
    kv_cache_bytes = 2 * L * batch_size * total_seq_len * d * bytes_per_param

    # Peak memory
    peak_mem = model_bytes + kv_cache_bytes

    # --- FLOPs ---
    prefill_flops = 2 * N * prompt_length * batch_size
    decode_flops_per_token = 2 * N * batch_size
    decode_flops_total = decode_flops_per_token * gen_tokens
    total_flops = prefill_flops + decode_flops_total

    # --- Latency ---
    if precision == "fp32":
        peak_tflops = hw['tflops_fp32']
    else:
        peak_tflops = hw['tflops_fp16']
    peak_flops = peak_tflops * 1e12
    utilization = 0.5
    eff_flops = peak_flops * utilization

    mem_bw = hw['mem_bw_gbs'] * 1e9

    # Prefill is compute-bound
    prefill_time_compute = prefill_flops / eff_flops if eff_flops > 0 else 0

    # Decode is memory-bandwidth-bound
    decode_time_per_token_membw = model_bytes / mem_bw if mem_bw > 0 else 0
    decode_time_per_token_compute = decode_flops_per_token / eff_flops if eff_flops > 0 else 0
    decode_time_per_token = max(decode_time_per_token_membw, decode_time_per_token_compute)

    decode_time_total = decode_time_per_token * gen_tokens
    total_time = prefill_time_compute + decode_time_total

    tokens_per_sec = gen_tokens / decode_time_total if decode_time_total > 0 else 0

    # Arithmetic intensity
    arith_intensity = (2 * N) / (N * bytes_per_param) if bytes_per_param > 0 else 0

    # --- Display ---
    print()
    print_divider("=", color=C.BGREEN)
    print(f"  {C.BGREEN}{C.BOLD}  INFERENCE ESTIMATES{C.RESET}")
    print_divider("=", color=C.BGREEN)
    print()

    print(f"    {C.BOLD}Memory:{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Model weights:        {C.BCYAN}{_fmt_bytes(model_bytes)}{C.RESET}  ({precision})")
    print(f"    {C.DIM}|--{C.RESET} KV Cache:             {C.BCYAN}{_fmt_bytes(kv_cache_bytes)}{C.RESET}  ({total_seq_len} tokens)")
    print(f"    {C.DIM}|--{C.RESET} Peak memory:          {C.BYELLOW}{C.BOLD}{_fmt_bytes(peak_mem)}{C.RESET}")
    fits = peak_mem <= hw['vram_gb'] * 1024**3
    fit_icon = f"{C.BGREEN}YES [OK]{C.RESET}" if fits else f"{C.BRED}NO [!]{C.RESET}"
    print(f"    {C.DIM}`--{C.RESET} Fits in VRAM ({hw['vram_gb']}GB)? {fit_icon}")
    print()

    print(f"    {C.BOLD}Latency:{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Prefill ({prompt_length} tokens):  {C.BGREEN}{prefill_time_compute * 1000:.1f} ms{C.RESET}  {C.DIM}(compute-bound){C.RESET}")
    bottleneck = "memory-BW" if decode_time_per_token_membw >= decode_time_per_token_compute else "compute"
    print(f"    {C.DIM}|--{C.RESET} Decode per token:      {C.BGREEN}{decode_time_per_token * 1000:.2f} ms{C.RESET}  {C.DIM}({bottleneck}-bound){C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Decode total ({gen_tokens} tok): {C.BYELLOW}{decode_time_total * 1000:.0f} ms{C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} Total (TTFT + gen):    {C.BYELLOW}{C.BOLD}{total_time * 1000:.0f} ms{C.RESET}  ({_fmt_time(total_time)})")
    print()

    print(f"    {C.BOLD}Throughput:{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Tokens/second:         {C.BGREEN}{C.BOLD}{tokens_per_sec:,.1f}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Arithmetic intensity:  {C.BCYAN}{arith_intensity:.1f} FLOPs/byte{C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} Total FLOPs:           {C.BCYAN}{_fmt_num(total_flops)} FLOPs{C.RESET}")

    # KV cache per-token table
    print()
    print(f"    {C.BOLD}KV Cache Growth:{C.RESET}")
    print(f"    {C.DIM}{'Seq Length':<15} {'KV Cache Size':<15} {'% of VRAM':<12}{C.RESET}")
    print(f"    {C.DIM}{'-' * 42}{C.RESET}")
    for seq_pct in [0.25, 0.5, 0.75, 1.0]:
        seq = int(S * seq_pct)
        kv = 2 * L * batch_size * seq * d * bytes_per_param
        vram_pct = (kv / (hw['vram_gb'] * 1024**3)) * 100
        bar_w = 10
        filled = min(int(bar_w * vram_pct / 100), bar_w)
        if vram_pct < 50:
            bc = C.BGREEN
        elif vram_pct < 80:
            bc = C.BYELLOW
        else:
            bc = C.BRED
        bar = f"{bc}{'#' * filled}{C.DIM}{'.' * (bar_w - filled)}{C.RESET}"
        print(f"    {seq:<15,} {_fmt_bytes(kv):<15} {bar} {vram_pct:.1f}%")

    wait_for_enter()


def calc_flops():
    """FLOPs Calculator -- detailed per-layer breakdown."""
    clear_screen()
    print_header("FLOPS CALCULATOR")

    print(f"  {C.DIM}Detailed FLOPs breakdown per layer, per forward/backward pass.{C.RESET}")
    print()

    specs = _get_model_specs()
    params = _count_transformer_params(
        specs['vocab_size'], specs['d_model'], specs['num_layers'],
        specs['d_ff'], specs['num_heads']
    )
    _print_model_card(specs, params)

    batch_size = int(get_input("Batch size", 1))
    seq_length = int(get_input("Sequence length", specs['max_seq_length']))
    print()

    N = params['total']
    L = specs['num_layers']
    d = specs['d_model']
    h = specs['num_heads']
    d_k = specs['head_dim']
    d_ff_val = specs['d_ff']
    V = specs['vocab_size']
    B = batch_size
    S = seq_length

    # --- Detailed FLOPs per layer ---
    qkv_flops = 3 * 2 * B * S * d * d
    attn_score_flops = 2 * B * h * S * S * d_k
    attn_value_flops = 2 * B * h * S * S * d_k
    out_proj_flops = 2 * B * S * d * d

    total_attn_flops = qkv_flops + attn_score_flops + attn_value_flops + out_proj_flops

    ffn_up_flops = 2 * B * S * d * d_ff_val
    ffn_down_flops = 2 * B * S * d_ff_val * d
    total_ffn_flops = ffn_up_flops + ffn_down_flops

    per_layer_flops = total_attn_flops + total_ffn_flops
    all_layers_flops = per_layer_flops * L

    output_flops = 2 * B * S * d * V

    total_fwd = all_layers_flops + output_flops
    total_bwd = 2 * total_fwd
    total_step = total_fwd + total_bwd

    quick_estimate = 6 * N * B * S

    # --- Display ---
    print_divider("=", color=C.BYELLOW)
    print(f"  {C.BYELLOW}{C.BOLD}  PER-LAYER FLOPS BREAKDOWN (Forward Pass){C.RESET}")
    print_divider("=", color=C.BYELLOW)
    print()

    all_parts = [
        ("QKV Projections", qkv_flops),
        ("Attention Scores (Q@K^T)", attn_score_flops),
        ("Attention Values (A@V)", attn_value_flops),
        ("Output Projection (W_O)", out_proj_flops),
        ("FFN Up   (d->d_ff)", ffn_up_flops),
        ("FFN Down (d_ff->d)", ffn_down_flops),
    ]

    max_f = max(f for _, f in all_parts)
    for name, flops in all_parts:
        pct = flops / per_layer_flops * 100
        bar_w = 25
        filled = int(bar_w * flops / max_f)
        bar = f"{C.BCYAN}{'#' * filled}{C.DIM}{'.' * (bar_w - filled)}{C.RESET}"
        print(f"    {name:<28} {bar}  {C.BOLD}{_fmt_num(flops):>8}{C.RESET}  {C.DIM}({pct:.1f}%){C.RESET}")

    print(f"    {C.DIM}{'-' * 70}{C.RESET}")

    attn_pct = total_attn_flops / per_layer_flops * 100
    ffn_pct = total_ffn_flops / per_layer_flops * 100
    print(f"    {'Attention (total)':<28} {' ' * 25}  {C.BGREEN}{C.BOLD}{_fmt_num(total_attn_flops):>8}{C.RESET}  {C.DIM}({attn_pct:.1f}%){C.RESET}")
    print(f"    {'FFN (total)':<28} {' ' * 25}  {C.BGREEN}{C.BOLD}{_fmt_num(total_ffn_flops):>8}{C.RESET}  {C.DIM}({ffn_pct:.1f}%){C.RESET}")
    print(f"    {'Per Layer':<28} {' ' * 25}  {C.BYELLOW}{C.BOLD}{_fmt_num(per_layer_flops):>8}{C.RESET}")

    print()
    print_divider("=", color=C.BGREEN)
    print(f"  {C.BGREEN}{C.BOLD}  TOTAL FLOPS (batch={B}, seq={S}){C.RESET}")
    print_divider("=", color=C.BGREEN)
    print()
    print(f"    {C.DIM}|--{C.RESET} All layers (fwd):     {C.BCYAN}{all_layers_flops:>20.3e}{C.RESET}  ({_fmt_num(all_layers_flops)})")
    print(f"    {C.DIM}|--{C.RESET} Output projection:     {C.BCYAN}{output_flops:>20.3e}{C.RESET}  ({_fmt_num(output_flops)})")
    print(f"    {C.DIM}|--{C.RESET} Forward pass total:    {C.BGREEN}{total_fwd:>20.3e}{C.RESET}  ({_fmt_num(total_fwd)})")
    print(f"    {C.DIM}|--{C.RESET} Backward pass (~2x):   {C.BYELLOW}{total_bwd:>20.3e}{C.RESET}  ({_fmt_num(total_bwd)})")
    print(f"    {C.DIM}|--{C.RESET} Total per step (f+b):  {C.BRED}{C.BOLD}{total_step:>20.3e}{C.RESET}  ({_fmt_num(total_step)})")
    print(f"    {C.DIM}`--{C.RESET} Quick estimate (6NBS):  {C.DIM}{quick_estimate:>18.3e}{C.RESET}  ({_fmt_num(quick_estimate)})")

    # Attention vs FFN dominance
    print()
    print(f"    {C.BOLD}Attention vs FFN Dominance:{C.RESET}")
    attn_bar_w = 40
    attn_filled = int(attn_bar_w * attn_pct / 100)
    print(f"    Attn {C.BCYAN}{'#' * attn_filled}{C.RESET}{C.BMAGENTA}{'#' * (attn_bar_w - attn_filled)}{C.RESET} FFN")
    print(f"         {C.BCYAN}{attn_pct:.1f}%{C.RESET}{' ' * (attn_bar_w - 10)}{C.BMAGENTA}{ffn_pct:.1f}%{C.RESET}")

    # Quadratic attention warning
    if S > 2048:
        print()
        print(f"    {C.BRED}{C.BOLD}WARNING:{C.RESET} {C.BRED}At seq_length={S}, attention is O(S^2) = O({S*S:,}).{C.RESET}")
        print(f"    {C.DIM}   Consider Flash Attention or sliding window attention for long contexts.{C.RESET}")

    wait_for_enter()


def calc_memory():
    """Memory Calculator -- full training memory breakdown."""
    clear_screen()
    print_header("MEMORY CALCULATOR")

    print(f"  {C.DIM}Estimates total GPU memory needed for training.{C.RESET}")
    print(f"  {C.DIM}Covers: model weights, optimizer states, gradients, activations.{C.RESET}")
    print()

    specs = _get_model_specs()
    params = _count_transformer_params(
        specs['vocab_size'], specs['d_model'], specs['num_layers'],
        specs['d_ff'], specs['num_heads']
    )
    _print_model_card(specs, params)

    batch_size = int(get_input("Batch size", 8))
    seq_length = int(get_input("Sequence length", specs['max_seq_length']))
    precision = get_input("Training precision (fp32 / fp16-mixed)", "fp16-mixed")
    optimizer_type = get_input("Optimizer (adam / sgd)", "adam")
    print()

    N = params['total']
    L = specs['num_layers']
    d = specs['d_model']
    d_ff_val = specs['d_ff']
    B = batch_size
    S = seq_length

    # --- Model Weights ---
    if precision == "fp32":
        weight_bytes = N * 4
        grad_bytes = N * 4
    else:
        weight_bytes = N * 2
        grad_bytes = N * 2

    # --- Optimizer States ---
    if optimizer_type == "adam":
        if precision == "fp16-mixed":
            optim_bytes = N * 4 + N * 4 + N * 4  # master + m + v
        else:
            optim_bytes = N * 4 + N * 4  # m + v
        optim_label = "Adam (m+v+master)" if precision == "fp16-mixed" else "Adam (m+v)"
    else:
        optim_bytes = 0
        optim_label = "SGD (no states)"

    # --- Activations ---
    bytes_per_el = 4 if precision == "fp32" else 2

    attn_input = B * S * d * bytes_per_el
    qkv_act = 3 * B * S * d * bytes_per_el
    h = specs['num_heads']
    attn_weights_act = B * h * S * S * bytes_per_el
    ffn_act = B * S * d_ff_val * bytes_per_el
    ln_act = 2 * B * S * d * bytes_per_el
    residual_act = 2 * B * S * d * bytes_per_el

    per_layer_act = attn_input + qkv_act + attn_weights_act + ffn_act + ln_act + residual_act
    total_act = per_layer_act * L

    emb_act = B * S * d * bytes_per_el
    output_act = B * S * specs['vocab_size'] * bytes_per_el

    total_act += emb_act + output_act

    # --- Grand Total ---
    grand_total = weight_bytes + grad_bytes + optim_bytes + total_act

    # --- Display ---
    print_divider("=", color=C.BGREEN)
    print(f"  {C.BGREEN}{C.BOLD}  TRAINING MEMORY BREAKDOWN{C.RESET}")
    print_divider("=", color=C.BGREEN)
    print()

    components = [
        ("Model Weights", weight_bytes, C.BCYAN),
        ("Gradients", grad_bytes, C.BGREEN),
        (f"Optimizer ({optim_label})", optim_bytes, C.BYELLOW),
        ("Activations", total_act, C.BMAGENTA),
    ]

    max_comp = max(c[1] for c in components)
    for name, size, color in components:
        pct = size / grand_total * 100
        bar_w = 30
        filled = int(bar_w * size / max_comp)
        bar = f"{color}{'#' * filled}{C.DIM}{'.' * (bar_w - filled)}{C.RESET}"
        print(f"    {name:<30} {bar}  {C.BOLD}{_fmt_bytes(size):>10}{C.RESET}  {C.DIM}({pct:.1f}%){C.RESET}")

    print(f"    {C.DIM}{'-' * 75}{C.RESET}")
    print(f"    {'TOTAL':<30} {' ' * 30}  {C.BOLD}{C.BRED}{_fmt_bytes(grand_total):>10}{C.RESET}")

    # VRAM comparison
    print()
    print(f"    {C.BOLD}VRAM Requirements:{C.RESET}")
    vram_options = [8, 16, 24, 40, 48, 80]
    for vram in vram_options:
        vram_bytes = vram * 1024**3
        pct = (grand_total / vram_bytes) * 100
        fits = grand_total <= vram_bytes
        icon = f"{C.BGREEN}[OK]{C.RESET}" if fits else f"{C.BRED}[!!]{C.RESET}"
        bar_w = 20
        filled = min(int(bar_w * pct / 100), bar_w)
        if pct < 70:
            bar_color = C.BGREEN
        elif pct < 95:
            bar_color = C.BYELLOW
        else:
            bar_color = C.BRED
        bar = f"{bar_color}{'#' * filled}{C.DIM}{'.' * (bar_w - filled)}{C.RESET}"
        print(f"    {icon} {vram:>3}GB  {bar}  {pct:>5.1f}%")

    # Activation breakdown
    print()
    print(f"    {C.BOLD}Activation Memory Breakdown (per layer):{C.RESET}")
    act_parts = [
        ("Attention input", attn_input),
        ("QKV projections", qkv_act),
        ("Attention weights (S^2)", attn_weights_act),
        ("FFN intermediate", ffn_act),
        ("Layer norms", ln_act),
        ("Residuals", residual_act),
    ]
    for name, size in act_parts:
        pct = size / per_layer_act * 100
        print(f"    {C.DIM}|--{C.RESET} {name:<25} {_fmt_bytes(size):>10}  {C.DIM}({pct:.1f}%){C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} {'Per layer total':<25} {C.BOLD}{_fmt_bytes(per_layer_act):>10}{C.RESET}")

    # Gradient checkpointing savings
    print()
    gc_savings = total_act - (per_layer_act * 2 + emb_act + output_act)
    if gc_savings > 0:
        print(f"    {C.BOLD}Gradient Checkpointing:{C.RESET}")
        print(f"    {C.DIM}|--{C.RESET} Without:  {C.BRED}{_fmt_bytes(total_act)}{C.RESET} activations")
        gc_act = per_layer_act * 2 + emb_act + output_act
        print(f"    {C.DIM}|--{C.RESET} With:     {C.BGREEN}{_fmt_bytes(gc_act)}{C.RESET} activations")
        print(f"    {C.DIM}`--{C.RESET} Savings:  {C.BCYAN}{_fmt_bytes(gc_savings)}{C.RESET} ({gc_savings / total_act * 100:.0f}% reduction)")

    wait_for_enter()


def calc_scaling_laws():
    """Scaling Laws Explorer -- Chinchilla & Kaplan scaling laws."""
    clear_screen()
    print_header("SCALING LAWS EXPLORER")

    print(f"  {C.DIM}Explore neural scaling laws that predict how loss scales with{C.RESET}")
    print(f"  {C.DIM}model size (N), dataset size (D), and compute budget (C).{C.RESET}")
    print()
    print(f"  {C.BOLD}Select a scaling law:{C.RESET}")
    print()
    print(f"    {C.BOLD}[1]{C.RESET}  Chinchilla Scaling Law  {C.DIM}(Hoffmann et al., 2022){C.RESET}")
    print(f"    {C.BOLD}[2]{C.RESET}  Kaplan Scaling Law      {C.DIM}(Kaplan et al., 2020){C.RESET}")
    print(f"    {C.BOLD}[3]{C.RESET}  Loss Prediction         {C.DIM}(Predict loss from N and D){C.RESET}")
    print(f"    {C.BOLD}[4]{C.RESET}  Model Comparison Table  {C.DIM}(Compare famous models){C.RESET}")
    print()
    choice = get_input("Select", "1")

    if choice == "1":
        # --- Chinchilla Scaling Law ---
        print()
        print_divider("=", color=C.BYELLOW)
        print(f"  {C.BYELLOW}{C.BOLD}  CHINCHILLA SCALING LAW{C.RESET}")
        print_divider("=", color=C.BYELLOW)
        print()
        print(f"  {C.DIM}Key insight: For compute-optimal training, model size N and{C.RESET}")
        print(f"  {C.DIM}dataset size D should scale equally with compute budget C.{C.RESET}")
        print()
        print(f"  {C.BCYAN}Optimal model:  N_opt = 0.0592 * C^0.50{C.RESET}")
        print(f"  {C.BCYAN}Optimal data:   D_opt = 1.776 * N_opt ~ 20 * N_opt{C.RESET}")
        print(f"  {C.BCYAN}Rule of thumb:  Train on ~20 tokens per parameter{C.RESET}")
        print()

        print(f"    {C.BOLD}Compute-Optimal Configurations:{C.RESET}")
        print(f"    {C.DIM}{'Compute (FLOPs)':<20} {'Optimal N':<15} {'Optimal D':<15} {'Tokens/Param':<15}{C.RESET}")
        print(f"    {C.DIM}{'-' * 65}{C.RESET}")

        for c_exp in [18, 19, 20, 21, 22, 23, 24, 25, 26]:
            C_budget = 10 ** c_exp
            N_opt = 0.0592 * (C_budget ** 0.50)
            D_opt = C_budget / (6 * N_opt)
            ratio = D_opt / N_opt
            print(f"    {C.CYAN}10^{c_exp:<17}{C.RESET} {_fmt_num(N_opt):<15} {_fmt_num(D_opt):<15} {ratio:>7.0f}x")

        print()
        print(f"  {C.BOLD}Your model:{C.RESET}")
        your_n = int(float(get_input("Enter your model's parameter count (e.g. 1e9)", 1e9)))
        chinchilla_d = 20 * your_n
        chinchilla_c = 6 * your_n * chinchilla_d
        print()
        print(f"    {C.DIM}|--{C.RESET} Your N:                  {C.BCYAN}{_fmt_num(your_n)}{C.RESET}")
        print(f"    {C.DIM}|--{C.RESET} Chinchilla-optimal D:    {C.BGREEN}{_fmt_num(chinchilla_d)}{C.RESET} tokens")
        print(f"    {C.DIM}|--{C.RESET} Required compute (6ND):  {C.BYELLOW}{chinchilla_c:.2e}{C.RESET} FLOPs")
        print(f"    {C.DIM}`--{C.RESET} Tokens-per-param ratio:  {C.BMAGENTA}20x{C.RESET}")

    elif choice == "2":
        # --- Kaplan Scaling Law ---
        print()
        print_divider("=", color=C.BYELLOW)
        print(f"  {C.BYELLOW}{C.BOLD}  KAPLAN SCALING LAW{C.RESET}")
        print_divider("=", color=C.BYELLOW)
        print()
        print(f"  {C.DIM}From 'Scaling Laws for Neural Language Models' (Kaplan et al., 2020){C.RESET}")
        print()
        print(f"  {C.BCYAN}L(N) = (N_c / N)^0.076     {C.DIM}(loss vs model size){C.RESET}")
        print(f"  {C.BCYAN}L(D) = (D_c / D)^0.095     {C.DIM}(loss vs dataset size){C.RESET}")
        print(f"  {C.BCYAN}L(C) = (C_c / C)^0.050     {C.DIM}(loss vs compute){C.RESET}")
        print()
        print(f"  {C.DIM}Where N_c ~ 8.8e13, D_c ~ 5.4e13, C_c ~ 3.1e8{C.RESET}")
        print()

        alpha_N = 0.076
        Nc = 8.8e13

        print(f"    {C.BOLD}Predicted Loss by Model Size:{C.RESET}")
        print(f"    {C.DIM}{'Parameters':<15} {'Predicted Loss':<18} {'Visualization':<30}{C.RESET}")
        print(f"    {C.DIM}{'-' * 55}{C.RESET}")

        for n_exp in [6, 7, 8, 9, 10, 11, 12, 13]:
            n = 10 ** n_exp
            loss = (Nc / n) ** alpha_N
            bar_w = 25
            fill_pct = max(0, min(1, (loss - 1.0) / 3.0))
            filled = int(bar_w * fill_pct)
            if loss < 2.0:
                loss_color = C.BGREEN
            elif loss < 3.0:
                loss_color = C.BYELLOW
            else:
                loss_color = C.BRED
            bar = f"{loss_color}{'#' * filled}{C.DIM}{'.' * (bar_w - filled)}{C.RESET}"
            print(f"    {_fmt_num(n):<15} {loss_color}{loss:<18.4f}{C.RESET} {bar}")

    elif choice == "3":
        # --- Loss Prediction ---
        print()
        print_divider("=", color=C.BYELLOW)
        print(f"  {C.BYELLOW}{C.BOLD}  LOSS PREDICTION{C.RESET}")
        print_divider("=", color=C.BYELLOW)
        print()
        print(f"  {C.DIM}Uses the combined scaling law:{C.RESET}")
        print(f"  {C.BCYAN}L(N, D) = E + A/N^a + B/D^b{C.RESET}")
        print(f"  {C.DIM}E=1.69, A=406.4, B=410.7, a=0.34, b=0.28{C.RESET}")
        print(f"  {C.DIM}(Hoffmann et al., 2022 -- Chinchilla paper){C.RESET}")
        print()

        E = 1.69
        A_coeff = 406.4
        B_coeff = 410.7
        alpha = 0.34
        beta = 0.28

        your_n = int(float(get_input("Model parameters (N)", 1e9)))
        your_d = int(float(get_input("Training tokens (D)", 20e9)))

        loss = E + A_coeff / (your_n ** alpha) + B_coeff / (your_d ** beta)
        compute = 6 * your_n * your_d

        chinchilla_d = 20 * your_n
        optimal_loss = E + A_coeff / (your_n ** alpha) + B_coeff / (chinchilla_d ** beta)

        print()
        print_divider("=", color=C.BGREEN)
        print(f"  {C.BGREEN}{C.BOLD}  PREDICTED LOSS{C.RESET}")
        print_divider("=", color=C.BGREEN)
        print()
        print(f"    {C.DIM}|--{C.RESET} Model size (N):          {C.BCYAN}{_fmt_num(your_n)}{C.RESET}")
        print(f"    {C.DIM}|--{C.RESET} Dataset size (D):        {C.BCYAN}{_fmt_num(your_d)}{C.RESET}")
        print(f"    {C.DIM}|--{C.RESET} Tokens-per-param:        {C.BCYAN}{your_d / your_n:.1f}x{C.RESET}")
        print(f"    {C.DIM}|--{C.RESET} Compute (6ND):           {C.BCYAN}{compute:.2e}{C.RESET} FLOPs")
        print(f"    {C.DIM}|--{C.RESET} {C.BOLD}Predicted loss:{C.RESET}           {C.BYELLOW}{C.BOLD}{loss:.4f}{C.RESET}")
        print(f"    {C.DIM}|--{C.RESET} Chinchilla-optimal loss: {C.BGREEN}{optimal_loss:.4f}{C.RESET}  (at D={_fmt_num(chinchilla_d)})")

        data_ratio = your_d / chinchilla_d
        if data_ratio < 0.8:
            status = f"{C.BRED}UNDER-TRAINED{C.RESET} -- need more data"
        elif data_ratio > 2.0:
            status = f"{C.BYELLOW}OVER-TRAINED{C.RESET} -- could use a larger model"
        else:
            status = f"{C.BGREEN}NEAR OPTIMAL{C.RESET}"
        print(f"    {C.DIM}`--{C.RESET} Status:                  {status}  {C.DIM}({data_ratio:.1f}x Chinchilla D){C.RESET}")

        perplexity = math.exp(loss)
        print()
        print(f"    {C.BOLD}Predicted Perplexity:{C.RESET}  {C.BMAGENTA}{C.BOLD}{perplexity:.1f}{C.RESET}  {C.DIM}(e^loss){C.RESET}")

    elif choice == "4":
        # --- Famous Model Comparison ---
        print()
        print_divider("=", color=C.BYELLOW)
        print(f"  {C.BYELLOW}{C.BOLD}  FAMOUS MODEL COMPARISON{C.RESET}")
        print_divider("=", color=C.BYELLOW)
        print()

        models = [
            ("GPT-2 Small",     117e6,    768,   12,  12,  3072,   1024,   "~40B?"),
            ("GPT-2 Medium",    345e6,    1024,  24,  16,  4096,   1024,   "~40B?"),
            ("GPT-2 Large",     774e6,    1280,  36,  20,  5120,   1024,   "~40B?"),
            ("GPT-2 XL",        1.5e9,    1600,  48,  25,  6400,   1024,   "~40B?"),
            ("GPT-3 Small",     125e6,    768,   12,  12,  3072,   2048,   "300B"),
            ("GPT-3 Medium",    350e6,    1024,  24,  16,  4096,   2048,   "300B"),
            ("GPT-3 Large",     760e6,    1536,  24,  16,  6144,   2048,   "300B"),
            ("GPT-3 XL",        1.3e9,    2048,  24,  32,  8192,   2048,   "300B"),
            ("GPT-3 (175B)",    175e9,    12288, 96,  96,  49152,  2048,   "300B"),
            ("LLaMA-7B",        7e9,      4096,  32,  32,  11008,  2048,   "1T"),
            ("LLaMA-13B",       13e9,     5120,  40,  40,  13824,  2048,   "1T"),
            ("LLaMA-65B",       65e9,     8192,  80,  64,  22016,  2048,   "1.4T"),
            ("Chinchilla-70B",  70e9,     8192,  80,  64,  22016,  2048,   "1.4T"),
            ("Omnicide (ours)", 60e3,     64,    2,   4,   256,    64,     "~3K"),
        ]

        print(f"    {C.BOLD}{'Model':<20} {'Params':<12} {'d_model':<9} {'Layers':<8} {'Heads':<7} {'Context':<9} {'Tokens':<8}{C.RESET}")
        print(f"    {C.DIM}{'-' * 80}{C.RESET}")

        for name, n_params, d, layers, heads, d_ff_val, ctx, tokens in models:
            if "Omnicide" in name:
                color = C.BRED + C.BOLD
            elif n_params >= 10e9:
                color = C.BYELLOW
            else:
                color = C.RESET
            print(f"    {color}{name:<20}{C.RESET} {_fmt_num(n_params):<12} {d:<9} {layers:<8} {heads:<7} {ctx:<9} {tokens:<8}")

        print()
        E, A_c, B_c, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
        print(f"    {C.BOLD}Chinchilla Optimality Check:{C.RESET}")
        checks = [
            ("GPT-3 (175B)", 175e9, 300e9),
            ("LLaMA-7B", 7e9, 1e12),
            ("LLaMA-65B", 65e9, 1.4e12),
            ("Chinchilla-70B", 70e9, 1.4e12),
        ]
        for name, n, d in checks:
            optimal_d = 20 * n
            ratio = d / optimal_d
            if ratio < 0.5:
                status = f"{C.BRED}under-trained{C.RESET}"
            elif ratio > 2.0:
                status = f"{C.BYELLOW}over-trained{C.RESET}"
            else:
                status = f"{C.BGREEN}~optimal{C.RESET}"
            print(f"    {C.DIM}|--{C.RESET} {name:<20} D/N={d/n:>6.0f}x (ideal~20x)  {status}")

    wait_for_enter()


def calc_compute_optimal():
    """Compute-Optimal Planner -- find ideal N and D for a given budget."""
    clear_screen()
    print_header("COMPUTE-OPTIMAL PLANNER")

    print(f"  {C.DIM}Given a compute budget (in GPU-hours or FLOPs), find the{C.RESET}")
    print(f"  {C.DIM}optimal model size and dataset size for best performance.{C.RESET}")
    print()

    print(f"  {C.BOLD}How would you like to specify your budget?{C.RESET}")
    print()
    print(f"    {C.BOLD}[1]{C.RESET}  GPU-hours  {C.DIM}(I know how many GPU-hours I have){C.RESET}")
    print(f"    {C.BOLD}[2]{C.RESET}  FLOPs      {C.DIM}(I know my compute budget in FLOPs){C.RESET}")
    print(f"    {C.BOLD}[3]{C.RESET}  Dollar budget  {C.DIM}(I know how much I can spend){C.RESET}")
    print()
    budget_type = get_input("Budget type", "1")

    if budget_type == "1":
        gpu_hours = float(get_input("Total GPU-hours", 1000))
        print()
        hw = _select_hardware()
        precision = get_input("Precision (fp16/fp32)", "fp16")
        peak = hw['tflops_fp16'] if precision != 'fp32' else hw['tflops_fp32']
        utilization = float(get_input("GPU utilization (0-1)", 0.35))
        total_flops = gpu_hours * 3600 * peak * 1e12 * utilization
    elif budget_type == "3":
        budget_dollars = float(get_input("Budget ($)", 10000))
        print()
        hw = _select_hardware()
        cost_per_hr = float(get_input("Cost per GPU-hour ($)", 2.50))
        precision = get_input("Precision (fp16/fp32)", "fp16")
        peak = hw['tflops_fp16'] if precision != 'fp32' else hw['tflops_fp32']
        utilization = float(get_input("GPU utilization (0-1)", 0.35))
        gpu_hours = budget_dollars / cost_per_hr
        total_flops = gpu_hours * 3600 * peak * 1e12 * utilization
    else:
        total_flops = float(get_input("Total compute budget (FLOPs, e.g. 1e21)", 1e21))

    C_budget = total_flops

    # Chinchilla optimal: C = 6ND, D_opt ~ 20N -> C = 120N^2 -> N = sqrt(C/120)
    N_opt = math.sqrt(C_budget / 120)
    D_opt = C_budget / (6 * N_opt)
    ratio = D_opt / N_opt

    # Predicted loss (Chinchilla parametric)
    E, A_coeff, B_coeff, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28
    predicted_loss = E + A_coeff / (N_opt ** alpha) + B_coeff / (D_opt ** beta)
    perplexity = math.exp(predicted_loss)

    def suggest_arch(n):
        if n < 50e6:
            return 512, 8, 6, 2048
        elif n < 200e6:
            return 768, 12, 12, 3072
        elif n < 500e6:
            return 1024, 16, 24, 4096
        elif n < 2e9:
            return 2048, 16, 24, 8192
        elif n < 10e9:
            return 4096, 32, 32, 11008
        elif n < 30e9:
            return 5120, 40, 40, 13824
        elif n < 100e9:
            return 8192, 64, 80, 22016
        else:
            return 12288, 96, 96, 49152

    d_model, num_heads, num_layers, d_ff_val = suggest_arch(N_opt)

    print()
    print_divider("=", color=C.BGREEN)
    print(f"  {C.BGREEN}{C.BOLD}  COMPUTE-OPTIMAL PLAN{C.RESET}")
    print_divider("=", color=C.BGREEN)
    print()

    print(f"    {C.BOLD}Budget:{C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} Compute:  {C.BYELLOW}{C.BOLD}{C_budget:.2e}{C.RESET} FLOPs  ({_fmt_num(C_budget)})")
    print()

    print(f"    {C.BOLD}Chinchilla-Optimal Configuration:{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Model size (N):       {C.BGREEN}{C.BOLD}{_fmt_num(N_opt)}{C.RESET}  ({N_opt:.2e})")
    print(f"    {C.DIM}|--{C.RESET} Dataset size (D):     {C.BGREEN}{C.BOLD}{_fmt_num(D_opt)}{C.RESET}  ({D_opt:.2e} tokens)")
    print(f"    {C.DIM}|--{C.RESET} Tokens/parameter:     {C.BCYAN}{ratio:.1f}x{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} Predicted loss:       {C.BYELLOW}{predicted_loss:.4f}{C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} Predicted perplexity: {C.BMAGENTA}{perplexity:.1f}{C.RESET}")
    print()

    print(f"    {C.BOLD}Suggested Architecture:{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} d_model:      {C.BCYAN}{d_model}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} num_heads:    {C.BCYAN}{num_heads}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} num_layers:   {C.BCYAN}{num_layers}{C.RESET}")
    print(f"    {C.DIM}|--{C.RESET} d_ff:         {C.BCYAN}{d_ff_val}{C.RESET}")
    print(f"    {C.DIM}`--{C.RESET} head_dim:     {C.BCYAN}{d_model // num_heads}{C.RESET}")
    print()

    # Alternative allocations
    print(f"    {C.BOLD}Alternative Allocations (same compute):{C.RESET}")
    print(f"    {C.DIM}{'Strategy':<25} {'Model (N)':<15} {'Data (D)':<15} {'Pred. Loss':<12}{C.RESET}")
    print(f"    {C.DIM}{'-' * 70}{C.RESET}")

    for label, n_mult in [("2x smaller model", 0.5), ("Chinchilla optimal", 1.0), ("2x larger model", 2.0), ("5x larger model", 5.0)]:
        n = N_opt * n_mult
        d = C_budget / (6 * n)
        loss = E + A_coeff / (n ** alpha) + B_coeff / (d ** beta)
        marker = "  <--" if n_mult == 1.0 else ""
        color = C.BGREEN if n_mult == 1.0 else C.RESET
        print(f"    {color}{label:<25}{C.RESET} {_fmt_num(n):<15} {_fmt_num(d):<15} {loss:<12.4f}{marker}")

    wait_for_enter()


def option_calculator_suite():
    """Main Calculator Suite menu."""
    while True:
        clear_screen()
        print_header("CALCULATOR SUITE")

        print(f"  {C.DIM}Transformer math calculators for training, inference, and scaling.{C.RESET}")
        print()

        print_menu_item("1", ">>", "Training Time Calculator",
                        "Estimate wall-clock training time from model specs & hardware")

        print_menu_item("2", ">>", "Inference Calculator",
                        "Latency, throughput, KV cache, and memory for inference")

        print_menu_item("3", ">>", "FLOPs Calculator",
                        "Detailed per-layer FLOP counts (attention vs FFN breakdown)")

        print_menu_item("4", ">>", "Memory Calculator",
                        "Training memory breakdown (weights, optimizer, activations, gradients)")

        print_menu_item("5", ">>", "Scaling Laws Explorer",
                        "Chinchilla & Kaplan scaling laws, loss prediction, model comparison")

        print_menu_item("6", ">>", "Compute-Optimal Planner",
                        "Find the ideal model size & dataset for your GPU budget")

        print_divider()
        print()
        print(f"  {C.DIM}[0]  Back to main menu{C.RESET}")
        print()

        choice = input(f"  {C.BCYAN}>{C.RESET} {C.BOLD}Calculator: {C.RESET}").strip()

        if choice == '1':
            calc_training_time()
        elif choice == '2':
            calc_inference()
        elif choice == '3':
            calc_flops()
        elif choice == '4':
            calc_memory()
        elif choice == '5':
            calc_scaling_laws()
        elif choice == '6':
            calc_compute_optimal()
        elif choice == '0' or choice.lower() in ('b', 'back', 'q'):
            return
        else:
            print_error("Invalid choice.")
            time.sleep(1)


# ==============================================================================
# UTILITY
# ==============================================================================
def _wrap_text(text, width):
    """Wrap text to fit within a given width."""
    lines = []
    for raw_line in text.split('\n'):
        if len(raw_line) <= width:
            lines.append(raw_line)
        else:
            while len(raw_line) > width:
                # Find the last space within the width
                break_idx = raw_line.rfind(' ', 0, width)
                if break_idx == -1:
                    break_idx = width
                lines.append(raw_line[:break_idx])
                raw_line = raw_line[break_idx:].lstrip()
            if raw_line:
                lines.append(raw_line)
    return lines


# ==============================================================================
# MAIN MENU
# ==============================================================================
def main_menu():
    """Display the main menu and return user's choice."""
    clear_screen()
    print_banner()

    print(f"  {C.BOLD}{C.BWHITE}Select an option:{C.RESET}")
    print()

    print_menu_item("1", "🧪", "Forward Pass Test",
                    "Run forward pass on both Encoder-Decoder and Decoder-Only models")

    print_menu_item("2", "🏋️", "Train a Model",
                    "Full pretraining pipeline with configurable hyperparameters")

    print_menu_item("3", "✨", "Quick Generate",
                    "Quick-train a small model and generate Shakespeare-style text")

    print_menu_item("4", "🔬", "Scalar Autograd Demo",
                    "Interactive demo of the Value engine (backpropagation on scalars)")

    print_menu_item("5", "🧬", "Tensor Autograd Demo",
                    "Matrix-level autograd — gradients through matmul, relu, and more")

    print_menu_item("6", "🔍", "Model Inspector",
                    "Inspect architecture details, parameter counts, and layer breakdown")

    print_menu_item("7", "📐", "Architecture Diagram",
                    "Visual diagram of the full Transformer architecture")

    print_menu_item("8", "🧮", "Math Reference",
                    "All the mathematical equations implemented in Omnicide")

    print_menu_item("9", "🔢", "Calculator Suite",
                    "Training time, inference, FLOPs, memory, and scaling laws")

    print_divider()
    print()
    print(f"  {C.DIM}[0]  Exit{C.RESET}")
    print()

    choice = input(f"  {C.BCYAN}❯{C.RESET} {C.BOLD}Enter your choice: {C.RESET}").strip()
    return choice


def main():
    """Main application entry point."""
    # Enable ANSI colors on Windows
    if os.name == 'nt':
        os.system('')  # Enables ANSI escape codes in Windows terminal

    while True:
        choice = main_menu()

        if choice == '1':
            option_forward_pass()
        elif choice == '2':
            option_train()
        elif choice == '3':
            option_generate()
        elif choice == '4':
            option_autograd_demo()
        elif choice == '5':
            option_tensor_autograd()
        elif choice == '6':
            option_inspect_model()
        elif choice == '7':
            option_architecture()
        elif choice == '8':
            option_math()
        elif choice == '9':
            option_calculator_suite()
        elif choice == '0' or choice.lower() in ('q', 'quit', 'exit'):
            clear_screen()
            print(f"""
{C.BRED}{C.BOLD}
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Thanks for using Omnicide!                                 ║
  ║                                                              ║
  ║   Built with 🧠 and NumPy.                                  ║
  ║   No frameworks were harmed in the making of this project.   ║
  ║                                                              ║
  ║   From scalar autograd to full Transformer pretraining —     ║
  ║   every gradient earned.                                     ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
{C.RESET}""")
            break
        else:
            print_error("Invalid choice. Please try again.")
            time.sleep(1)


if __name__ == "__main__":
    main()
