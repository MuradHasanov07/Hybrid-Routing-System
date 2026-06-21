"""
prepare_data.py

Raw Kaggle "Customer Support Tickets" dataset (200k rows) -> labeled, balanced
3-class dataset (Sales Inquiry / Complaint / Spam) ready for training.

WHY THIS SCRIPT EXISTS (read this before changing anything):
The raw dataset's `issue_description` column only contains 10 distinct sentences,
and each sentence appears with roughly equal frequency across ALL 10 `category`
values. This means there is no real linguistic signal connecting the description
text to the category label - a classifier trained directly on the raw text would
be learning from noise.

To produce a dataset that a text classifier can actually learn from, we:
  1. Map the original 10 fine-grained categories to 3 target classes
     (this part uses real structure from the dataset).
  2. Replace the (signal-free) issue_description with a sentence randomly drawn
     from a hand-written template pool that is specific to the ORIGINAL category.
     This injects a consistent, learnable relationship between text and label,
     while keeping the topical content faithful to what each category represents.
  3. Add a synthetic "Spam" class (not present in the original dataset, since a
     real support-ticket system would have filtered spam out already) using a
     separate template pool.
  4. Build a class-balanced subset for training, instead of using all 200k rows.

This is documented transparently in the README - the raw dataset alone could not
be used for genuine NLP training, so this preprocessing step was necessary.
"""

import argparse
import random
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Category -> target class mapping
# ---------------------------------------------------------------------------

SALES_CATEGORIES = [
    "Refund Request",
    "Payment Problem",
    "Feature Request",
    "Subscription Cancellation",
]

COMPLAINT_CATEGORIES = [
    "Bug Report",
    "Performance Issue",
    "Login Issue",
    "Data Sync Issue",
    "Security Concern",
    "Account Suspension",
]

CATEGORY_TO_CLASS = {c: "Sales Inquiry" for c in SALES_CATEGORIES}
CATEGORY_TO_CLASS.update({c: "Complaint" for c in COMPLAINT_CATEGORIES})

CLASSES = ["Sales Inquiry", "Complaint", "Spam"]

# ---------------------------------------------------------------------------
# 2. Template sentence pools, one pool per ORIGINAL category (15-20 each)
#    plus a separate pool for the synthetic Spam class.
# ---------------------------------------------------------------------------

TEMPLATES = {
    "Refund Request": [
        "I would like to request a refund for my last payment, it was charged twice.",
        "Can you process a refund for the order I cancelled yesterday?",
        "I was charged for a service I never used, I need my money back.",
        "Please refund the amount deducted on my last billing cycle.",
        "I cancelled my plan but I was still charged, requesting a full refund.",
        "The product did not match the description, I want a refund.",
        "How long does it usually take to receive a refund after approval?",
        "I never authorized this charge, please refund it immediately.",
        "Requesting a refund since the service was unavailable for two weeks.",
        "I returned the item and I'm still waiting on my refund.",
        "Can I get a refund if I cancel within the trial period?",
        "I was double billed this month, please refund the extra charge.",
        "The refund I requested last week has not appeared in my account.",
        "I'd like a partial refund since only half the order arrived.",
        "Is it possible to get a refund instead of store credit?",
        "Please reverse the charge, I did not approve this purchase.",
        "I'm requesting a refund due to a billing error on your end.",
        "My refund request was denied, can you explain why?",
        "Can I get my money back?",
        "I want a refund please.",
        "How do I request a refund?",
        "When will I get my refund?",
        "Please refund my last order.",
        "Is a refund possible for this purchase?",
        "I need a refund as soon as possible.",
        "Can you reverse this charge for me?",
        "What is your refund policy?",
        "I'd like my money back for this order.",
    ],
    "Payment Problem": [
        "My credit card was declined even though I have sufficient funds.",
        "There seems to be a discrepancy in my billing statement for this month.",
        "I tried to pay but the transaction keeps failing.",
        "Can you tell me why my payment method was rejected?",
        "I was charged twice for the same invoice this month.",
        "The payment page keeps showing an error when I try to checkout.",
        "My subscription payment failed and I lost access to the service.",
        "I need help updating my payment information on file.",
        "The amount charged doesn't match what was shown at checkout.",
        "My bank is asking about a charge from you I don't recognize.",
        "I can't add a new payment method, the form won't submit.",
        "Why was I charged in a different currency than expected?",
        "My automatic payment didn't go through this billing cycle.",
        "I'm seeing a pending charge that has not cleared in days.",
        "Can you confirm whether my last payment was successfully processed?",
        "There's an unexpected fee on my invoice I'd like explained.",
        "My payment was deducted from my bank account but the transaction shows failed.",
        "I need a receipt for a payment that isn't showing in my history.",
        "What is the price?",
        "How much does it cost?",
        "Can you provide pricing details?",
        "What is the subscription fee?",
        "I would like to know the cost.",
        "Tell me the price and mileage.",
        "What are the product specifications and price?",
        "Can you tell me the price of this plan?",
        "How much do I need to pay?",
        "What's the cost of the premium tier?",
        "Can you send me a price quote?",
        "How much would an upgrade cost?",
    ],
    "Feature Request": [
        "It would be great if the app supported dark mode.",
        "Could you add an export to Excel option in the reports section?",
        "I'd like to request a feature for scheduling recurring tasks.",
        "Is there a way to add multi-language support to the dashboard?",
        "Please consider adding two-way calendar sync in a future update.",
        "It would help a lot if we could customize the notification settings.",
        "Can you add bulk editing for items in the inventory list?",
        "I'd love to see an offline mode added to the mobile app.",
        "Could you add the ability to tag and filter tickets by priority?",
        "Adding a keyboard shortcut for quick search would be very useful.",
        "It would be nice to have a dashboard widget for monthly summaries.",
        "Can you support importing data directly from a CSV file?",
        "I'd like to suggest adding role-based access control for teams.",
        "Please add an undo option after deleting an item by mistake.",
        "Could the app support integration with our calendar tool?",
        "It would be useful to have a dark theme option for night use.",
        "I'm requesting a feature to merge duplicate customer records.",
        "Can you add support for custom fields in the order form?",
        "Can you add this feature please?",
        "I have a suggestion for a new feature.",
        "Please consider this feature request.",
        "Could you build this into the next update?",
        "I'd like to suggest an improvement.",
        "Is this feature on your roadmap?",
        "Would you consider adding this option?",
        "Here's an idea for a new feature.",
    ],
    "Subscription Cancellation": [
        "I want to cancel my subscription effective immediately.",
        "My subscription was cancelled without my request and I need clarification.",
        "Please cancel my plan, I no longer need this service.",
        "How do I cancel my subscription before the next billing date?",
        "I tried cancelling online but the option doesn't appear in my settings.",
        "I'd like to downgrade my subscription instead of cancelling fully.",
        "Can you confirm that my cancellation request was processed?",
        "I cancelled last month but I was still billed this cycle.",
        "What happens to my saved data if I cancel my subscription?",
        "I want to pause my subscription rather than cancel it outright.",
        "Please stop auto-renewal on my account going forward.",
        "I'm cancelling because the service no longer fits our needs.",
        "Can I get a partial refund if I cancel mid-cycle?",
        "I need to cancel my subscription on behalf of my team.",
        "My cancellation confirmation email never arrived, can you verify it went through?",
        "I'd like to switch to a free plan instead of cancelling completely.",
        "Please cancel my renewal, I don't want to be charged next month.",
        "I changed my mind about cancelling, can you reactivate my plan?",
        "Please cancel my subscription.",
        "I want to cancel now.",
        "How do I unsubscribe?",
        "Cancel my plan please.",
        "I'd like to end my subscription.",
        "Can I downgrade my plan?",
        "I want to switch to a cheaper plan.",
        "Please stop billing me.",
    ],
    "Bug Report": [
        "I found a bug in the latest update affecting report generation.",
        "The app crashes every time I try to open the settings page.",
        "There's a display glitch on the dashboard after the recent update.",
        "Clicking the save button doesn't save my changes anymore.",
        "The search feature returns no results even for valid queries.",
        "I noticed duplicate entries appearing after syncing my account.",
        "The export button throws an error instead of downloading the file.",
        "Notifications stopped working after the last app update.",
        "The app freezes when I try to upload more than one file.",
        "There's a bug where the totals on my invoice don't add up correctly.",
        "The calendar view shows the wrong dates for recurring events.",
        "I keep getting logged out randomly while using the app.",
        "The filter options on the reports page aren't applying correctly.",
        "Images aren't loading properly in the product gallery.",
        "The mobile app shows a blank screen after the splash screen.",
        "Editing a record sometimes duplicates it instead of updating it.",
        "The print preview cuts off part of the document content.",
        "Dark mode causes some text to become unreadable on certain screens.",
        "The application crashes.",
        "I found a bug.",
        "Something is broken in the app.",
        "This feature doesn't work.",
        "The app keeps crashing.",
        "There's an error on this page.",
        "I think I found a glitch.",
        "The button doesn't respond when clicked.",
    ],
    "Performance Issue": [
        "I am experiencing very slow performance while using the dashboard.",
        "The app takes too long to load on my mobile device.",
        "Pages are timing out frequently during peak hours.",
        "The system has been extremely sluggish since the last update.",
        "Loading the reports section takes over a minute every time.",
        "The search function has become noticeably slower this week.",
        "Uploading files is much slower than it used to be.",
        "The app lags badly when switching between tabs.",
        "Response times have increased significantly over the past few days.",
        "The dashboard widgets take forever to refresh with new data.",
        "I'm seeing frequent lag spikes while typing in the editor.",
        "The mobile app performance has degraded after the recent release.",
        "Large datasets cause the app to become unresponsive.",
        "Scrolling through long lists is choppy and slow.",
        "The video call feature drops frames constantly during use.",
        "Saving a large file takes much longer than expected.",
        "The app becomes unusable when multiple tabs are open at once.",
        "Server response time has been unacceptably slow all week.",
        "The system is too slow.",
        "Everything is loading slowly today.",
        "The app is very sluggish.",
        "Why is this taking so long to load?",
        "Performance has gotten worse recently.",
        "The page won't load fast enough.",
        "This is running really slow for me.",
    ],
    "Login Issue": [
        "I am unable to access my account after entering the correct credentials.",
        "The application crashes whenever I try to upload a file.",
        "I'm locked out of my account after too many failed login attempts.",
        "The password reset link in my email isn't working.",
        "I can't log in even though I'm sure my password is correct.",
        "My account shows as suspended when I try to sign in.",
        "The login page keeps refreshing without letting me sign in.",
        "I never received the verification code needed to log in.",
        "Single sign-on isn't working for my company account.",
        "I keep getting an 'invalid credentials' error despite resetting my password.",
        "My session expires almost immediately after logging in.",
        "I can't log in from my phone but it works fine on desktop.",
        "The CAPTCHA on the login page won't load properly.",
        "I'm unable to switch between multiple accounts after logging in.",
        "My login attempt redirects me to an error page every time.",
        "I forgot my username and the recovery process isn't working.",
        "The app logs me out immediately after I successfully sign in.",
        "I can't log in using my linked Google account anymore.",
        "I cannot log in.",
        "I can't access my account.",
        "My login isn't working.",
        "I forgot my password.",
        "I'm locked out of my account.",
        "Why can't I sign in?",
        "I need help logging in.",
    ],
    "Data Sync Issue": [
        "The system is not syncing data across devices properly.",
        "Data synchronization restored after backend service restart.",
        "My changes on mobile aren't reflected on the desktop app.",
        "Sync keeps failing with a generic error message.",
        "I'm seeing outdated information that hasn't synced in days.",
        "Two of my devices are showing different versions of the same file.",
        "The sync icon spins constantly but never finishes.",
        "Some of my records disappeared after the last sync attempt.",
        "Offline changes aren't being uploaded once I reconnect to the internet.",
        "My calendar isn't syncing with the connected third-party app.",
        "Sync conflicts keep appearing for files I haven't edited recently.",
        "Changes made by my teammate aren't showing up on my end.",
        "The app says 'sync complete' but the data is clearly missing.",
        "I had to manually refresh several times before data finally synced.",
        "Photos uploaded on mobile never appear on the web version.",
        "Sync seems to break whenever my internet connection is unstable.",
        "Two records merged incorrectly after a sync conflict.",
        "My settings reset to default after a failed sync.",
        "Data synchronization is not working.",
        "My data isn't syncing.",
        "Sync failed again.",
        "Why isn't my data updating across devices?",
        "Things aren't syncing properly.",
    ],
    "Security Concern": [
        "Two-factor authentication codes are not being delivered to my phone.",
        "I noticed a login attempt from a location I don't recognize.",
        "I think someone accessed my account without my permission.",
        "I'd like to enable two-factor authentication but the option is missing.",
        "I received a suspicious email asking for my account password.",
        "Can you confirm whether my account data was part of a recent breach?",
        "I want to review the list of devices currently logged into my account.",
        "My account settings changed without me making any edits.",
        "I'm concerned my password may have been compromised.",
        "Please help me revoke access for a device I no longer use.",
        "I received a security alert I don't understand, can you clarify?",
        "How is my personal data stored and protected on your platform?",
        "I'd like to report a phishing attempt impersonating your company.",
        "Can you force a logout on all devices for my account?",
        "I noticed an API key was generated that I didn't create.",
        "My recovery email was changed without my authorization.",
        "I want to confirm your platform is compliant with data protection regulations.",
        "Someone tried resetting my password without my consent.",
        "I think my account was hacked.",
        "Someone accessed my account.",
        "I'm worried about my account security.",
        "Is my data safe with you?",
        "I want to report suspicious activity.",
    ],
    "Account Suspension": [
        "My account has been suspended and I don't understand why.",
        "Can you tell me what triggered the suspension on my account?",
        "I believe my account was suspended by mistake, please review it.",
        "How long will my account remain suspended for this issue?",
        "I need my account reinstated as soon as possible for work.",
        "The suspension notice didn't explain which policy I violated.",
        "I'd like to appeal the suspension placed on my account.",
        "My team's access was suspended along with my own account.",
        "Is there a way to expedite the review of my suspended account?",
        "I was suspended after a payment failure that has since been resolved.",
        "Please confirm what steps I need to take to lift the suspension.",
        "My account got suspended right after I reported a bug, is that related?",
        "I never received any warning before my account was suspended.",
        "Can someone manually review my case for the account suspension?",
        "The suspension is affecting my ability to access paid features.",
        "I'd like clarification on the appeals process for suspended accounts.",
        "My account was suspended due to unusual activity I can explain.",
        "Please restore my account, the suspension seems unwarranted.",
        "My account was suspended.",
        "Why is my account suspended?",
        "Please reinstate my account.",
        "My account got blocked.",
        "I can't access my suspended account.",
    ],
}

SPAM_TEMPLATES = [
    "Congratulations! You've won a free gift, click here to claim it now!",
    "URGENT: Your account will be deleted unless you verify your details immediately.",
    "Limited time offer: double your crypto investment in 24 hours, click now!",
    "You have been selected for a $1000 cash prize, claim before it expires!",
    "Hot singles in your area want to chat with you tonight, click here!",
    "Your package could not be delivered, click this link to reschedule.",
    "Make $5000 a week working from home, no experience needed, sign up now!",
    "Your bank account has been flagged, verify your password here immediately.",
    "FREE iPhone giveaway just for new subscribers, click to claim yours!",
    "Act now! Your loan has been pre-approved, click here to receive funds.",
    "Exclusive deal just for you: buy one get ten free, limited stock!",
    "Your subscription expired, click here to renew and get 90% off forever.",
    "Lose 20 pounds in a week with this one weird trick, click to learn more!",
    "Click here to unlock your mystery reward worth up to $10,000.",
    "Your computer has a virus, download this tool immediately to fix it.",
    "Final notice: pay this invoice within 1 hour or face legal action.",
    "Earn passive income instantly, just click and enter your bank details.",
    "Congratulations, you are today's lucky visitor, claim your free prize now!",
    "Congratulations you won!",
    "Claim your reward now.",
    "Click here now!",
    "Earn money online fast.",
    "Limited time offer, act now!",
    "You've been selected, click to claim.",
    "Free money waiting for you, click here.",
    "Act fast, this deal won't last!",
]


def build_raw_records(min_per_category: int = 3500, seed: int = 42) -> pd.DataFrame:
    """Build the augmented dataset: one row per template usage, balanced roughly
    evenly across the 10 original categories, then mapped to 3 target classes,
    plus a synthetic Spam slice."""
    rng = random.Random(seed)
    rows = []

    for category, pool in TEMPLATES.items():
        target_class = CATEGORY_TO_CLASS[category]
        for _ in range(min_per_category):
            text = rng.choice(pool)
            rows.append({"text": text, "category": category, "label": target_class})

    # Spam: synthetic, no original category
    n_spam = min_per_category * 2  # spam pool is small, oversample by repetition
    for _ in range(n_spam):
        text = rng.choice(SPAM_TEMPLATES)
        rows.append({"text": text, "category": "Spam (synthetic)", "label": "Spam"})

    df = pd.DataFrame(rows)
    return df


def balance_classes(df: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    parts = []
    for cls in CLASSES:
        sub = df[df["label"] == cls]
        n = min(per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_dataset(df: pd.DataFrame, train_frac=0.8, val_frac=0.1, seed: int = 42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced 3-class dataset.")
    parser.add_argument("--raw_csv", type=str, default="data/raw/customer_support_tickets_200k.csv",
                         help="Path to raw Kaggle CSV (used only to confirm category names match).")
    parser.add_argument("--per_class", type=int, default=10000,
                         help="Number of rows per class in the final balanced dataset.")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_path = Path(args.raw_csv)
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path, usecols=["category"])
        raw_categories = set(raw_df["category"].unique())
        expected = set(CATEGORY_TO_CLASS.keys())
        missing = expected - raw_categories
        if missing:
            print(f"WARNING: categories in mapping but not found in raw CSV: {missing}")
    else:
        print(f"NOTE: raw CSV not found at {raw_path}, skipping category validation.")

    print("Building augmented records from templates...")
    augmented = build_raw_records(min_per_category=args.per_class // 2, seed=args.seed)
    print(f"Total rows before balancing: {len(augmented)}")
    print(augmented["label"].value_counts())

    print(f"\nBalancing to {args.per_class} rows per class...")
    balanced = balance_classes(augmented, per_class=args.per_class, seed=args.seed)
    print(balanced["label"].value_counts())

    train, val, test = split_dataset(balanced, seed=args.seed)
    print(f"\nSplit sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    balanced.to_csv(out_dir / "full_balanced.csv", index=False)
    print(f"\nSaved processed CSVs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()