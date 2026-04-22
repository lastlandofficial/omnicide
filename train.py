"""
================================================================================
 PRETRAINING LOOP — Teaching a Transformer to Predict Text
================================================================================
This script ties everything together into a complete training pipeline:

    1. Tokenize the training text (character-level)
    2. Initialize the model, loss function, and optimizer
    3. Train for N iterations:
       - Sample a batch of (input, target) sequences
       - Forward pass: compute logits
       - Compute loss (cross-entropy)
       - Backward pass: compute all gradients via autograd
       - Clip gradients (for stability)
       - Update weights (Adam optimizer)
    4. Generate sample text to verify learning

================================================================================
WHAT THE MODEL LEARNS:
================================================================================
By predicting the next character millions of times, the model learns:
- Which characters commonly follow which (e.g., 'q' → 'u')
- Word-level patterns (e.g., 'th' → 'e', 'the')
- Basic grammar (e.g., spaces between words, punctuation patterns)
- Character-level "spelling" of common words

With a tiny model like ours (d_model=64, 2 layers), it will learn basic
character bigram/trigram patterns. Larger models learn deeper patterns.
================================================================================
"""
import numpy as np
import time

from config import TransformerConfig
from model import DecoderOnlyTransformer
from loss import CrossEntropyLoss
from optim import Adam, clip_grad_norm
from data import CharTokenizer, DataLoader


# ==============================================================================
# TRAINING TEXT — Shakespeare's Coriolanus (excerpt)
# ==============================================================================
# We use a substantial chunk of Shakespeare so the model has enough patterns
# to learn. Character-level modeling needs repetitive patterns in the text.
TRAINING_TEXT = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report for't, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country, he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city is risen:
why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?

Second Citizen:
Worthy Menenius Agrippa; one that hath always loved the people.

First Citizen:
He's one honest enough: would all the rest were so!

What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have
had inkling this fortnight what we intend to do,
which now we'll show 'em in deeds. They say poor
suitors have strong breaths: they shall know we
have strong arms too.

Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?

First Citizen:
We cannot, sir, we are undone already.

I tell you, friends, most charitable care
Have the patricians of you. For your wants,
Your suffering in this dearth, you may as well
Strike at the heaven with your staves as lift them
Against the Roman state, whose course will on
The way it takes, cracking ten thousand curbs
Of more strong link asunder than can ever
Appear in your impediment. For the dearth,
The gods, not the patricians, make it, and
Your knees to them, not arms, must help. Alack,
You are transported by calamity. Thither,
Where more attends you, and you slander
The helms o' the state, who care for you like fathers,
When you curse them as enemies.

First Citizen:
Care for us! True, indeed! They ne'er cared for us
yet: suffer us to famish, and their store-houses
crammed with grain; make edicts for usury, to
support usurers; repeal daily any wholesome act
established against the rich, and provide more
piercing statutes daily, to chain up and restrain
the poor. If the wars eat us not up, they will; and
there's all the love they bear us.
"""


def main():
    """
    =========================================================================
    MAIN TRAINING FUNCTION
    =========================================================================
    """
    print("=" * 70)
    print("  TRANSFORMER PRETRAINING FROM SCRATCH (Pure NumPy)")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # ==================================================================
    # STEP 1: TOKENIZATION
    # ==================================================================
    print("\n--- Step 1: Tokenization ---")
    tokenizer = CharTokenizer()
    tokenizer.fit(TRAINING_TEXT)

    encoded_text = tokenizer.encode(TRAINING_TEXT)
    print(f"Sample encoding: '{TRAINING_TEXT[:30]}...'")
    print(f"  -> {encoded_text[:30]}...")

    # ==================================================================
    # STEP 2: MODEL INITIALIZATION
    # ==================================================================
    print("\n--- Step 2: Model Initialization ---")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,          # Small embedding dim (GPT-2 uses 768)
        num_heads=4,          # 4 attention heads (GPT-2 uses 12)
        num_layers=2,         # 2 transformer blocks (GPT-2 uses 12)
        d_ff=256,             # FFN hidden dim (GPT-2 uses 3072)
        max_seq_length=64,    # Max context window
    )

    model = DecoderOnlyTransformer(config)

    # Count parameters
    total_params = sum(p.data.size for p in model.parameters())
    trainable_params = sum(p.data.size for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ==================================================================
    # STEP 3: LOSS FUNCTION & OPTIMIZER
    # ==================================================================
    print("\n--- Step 3: Loss & Optimizer ---")
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    print(f"Optimizer: Adam (lr={optimizer.lr}, b1={optimizer.beta1}, b2={optimizer.beta2})")

    # ==================================================================
    # STEP 4: DATA LOADER
    # ==================================================================
    print("\n--- Step 4: Data Loader ---")
    batch_size = 4
    seq_length = 32

    dataloader = DataLoader(encoded_text, batch_size=batch_size, seq_length=seq_length)

    # ==================================================================
    # STEP 5: TRAINING LOOP
    # ==================================================================
    num_steps = 1000
    print_every = 100
    max_grad_norm = 1.0

    print(f"\n--- Step 5: Training ({num_steps} steps) ---")
    print(f"Batch size: {batch_size}, Seq length: {seq_length}")
    print(f"Gradient clipping: max_norm={max_grad_norm}")
    print("-" * 70)

    losses = []
    start_time = time.time()

    for step in range(num_steps):
        step_start = time.time()

        # --- (a) Sample a batch ---
        X, Y = dataloader.get_batch()
        # X: (batch_size, seq_length) — input token IDs
        # Y: (batch_size, seq_length) — target token IDs (X shifted right by 1)

        # --- (b) Forward pass ---
        # Compute logits: the model's prediction scores for each vocab token
        logits = model.forward(X)
        # logits: (batch_size, seq_length, vocab_size)

        # --- (c) Compute loss ---
        # How wrong is the model? Cross-entropy measures this.
        loss = criterion(logits, Y)

        # --- (d) Zero gradients ---
        # IMPORTANT: Reset all gradients to 0 before backward pass.
        # Otherwise, gradients accumulate across steps, which is wrong.
        optimizer.zero_grad()

        # --- (e) Backward pass ---
        # THIS IS WHERE THE MAGIC HAPPENS:
        # The autograd engine traverses the computation graph in reverse,
        # computing ∂Loss/∂(parameter) for every learnable parameter
        # using the chain rule. This single call propagates gradients
        # from the loss all the way back through:
        #   loss → cross_entropy → fc_out → norm → blocks → ... → embedding
        loss.backward()

        # --- (f) Gradient clipping ---
        # Prevents exploding gradients that can destabilize training
        grad_norm = clip_grad_norm(model.parameters(), max_grad_norm)

        # --- (g) Optimizer step ---
        # Update all parameters using Adam:
        #   θ -= lr · m̂ / (√v̂ + ε)
        optimizer.step()

        # --- Logging ---
        loss_val = float(loss.data)
        losses.append(loss_val)
        step_time = time.time() - step_start

        if step % print_every == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            avg_loss = np.mean(losses[-print_every:]) if len(losses) >= print_every else np.mean(losses)
            print(
                f"Step {step:4d}/{num_steps} | "
                f"Loss: {loss_val:.4f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Grad Norm: {grad_norm:.4f} | "
                f"Step Time: {step_time:.2f}s | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # ==================================================================
    # STEP 6: RESULTS SUMMARY
    # ==================================================================
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final loss:    {losses[-1]:.4f}")
    print(f"Initial loss:  {losses[0]:.4f}")
    print(f"Loss decrease: {losses[0] - losses[-1]:.4f}")
    print(f"Total time:    {total_time:.1f} seconds")
    print(f"Expected initial loss (random): ~{np.log(tokenizer.vocab_size):.2f} "
          f"(-log(1/{tokenizer.vocab_size}))")

    # ==================================================================
    # STEP 7: TEXT GENERATION (Proof of Learning)
    # ==================================================================
    print("\n--- Step 7: Text Generation ---")
    print("Generating text from prompt...\n")

    # Encode a short prompt
    prompt = "First Citizen:"
    prompt_ids = np.array([tokenizer.encode(prompt)])  # (1, prompt_len)

    # Generate text
    generated_ids = model.generate(prompt_ids, max_new_tokens=150, temperature=0.8)
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated_text}'")
    print()


if __name__ == "__main__":
    main()
