/**
 * Display metadata for each model output class.
 * Order and keys must match training/skinlesion/config.py::CLASSES.
 */

export type Category = "malignant" | "precancerous" | "benign" | "non-lesion" | "invalid";

export interface ClassInfo {
  key: string;
  name: string;
  short: string;
  category: Category;
  description: string;
}

export const CLASSES: readonly ClassInfo[] = [
  {
    key: "akiec",
    name: "Actinic keratosis / intraepithelial carcinoma",
    short: "Actinic keratosis",
    category: "precancerous",
    description:
      "Rough, scaly patches caused by years of sun exposure. Considered precancerous — a small share progress to squamous cell carcinoma, so they are usually treated.",
  },
  {
    key: "bcc",
    name: "Basal cell carcinoma",
    short: "Basal cell carcinoma",
    category: "malignant",
    description:
      "The most common skin cancer. Often a pearly bump or a sore that won't heal. Grows slowly and rarely spreads, but needs treatment.",
  },
  {
    key: "bkl",
    name: "Benign keratosis-like lesion",
    short: "Benign keratosis",
    category: "benign",
    description:
      "Seborrheic keratoses, solar lentigines and lichen-planus-like keratoses. Common, harmless 'stuck-on' growths and age spots.",
  },
  {
    key: "df",
    name: "Dermatofibroma",
    short: "Dermatofibroma",
    category: "benign",
    description:
      "A firm, harmless nodule — often on the legs — that dimples inward when pinched. Frequently follows a minor injury or insect bite.",
  },
  {
    key: "mel",
    name: "Melanoma",
    short: "Melanoma",
    category: "malignant",
    description:
      "The most dangerous skin cancer, arising from pigment cells. Watch for Asymmetry, irregular Borders, multiple Colours, Diameter > 6 mm and Evolution.",
  },
  {
    key: "nv",
    name: "Melanocytic nevus",
    short: "Mole (nevus)",
    category: "benign",
    description:
      "An ordinary mole — a benign cluster of pigment cells. Most adults have 10–40. Changes in size, shape or colour are worth a check.",
  },
  {
    key: "vasc",
    name: "Vascular lesion",
    short: "Vascular lesion",
    category: "benign",
    description:
      "Cherry angiomas, angiokeratomas, pyogenic granulomas and haemorrhage. Red-to-purple lesions formed from blood vessels; almost always benign.",
  },
  {
    key: "other_skin",
    name: "Other skin condition (non-cancerous)",
    short: "Other skin condition",
    category: "non-lesion",
    description:
      "Rashes and inflammatory conditions such as eczema, dermatitis, hives, psoriasis, acne or insect bites — not a tumour-type lesion.",
  },
  {
    key: "not_skin",
    name: "Not a skin image",
    short: "Not skin",
    category: "invalid",
    description:
      "Everyday objects, scenes, animals and textures. The model was trained to recognise these so it can refuse to diagnose them.",
  },
] as const;

export const CATEGORY_LABEL: Record<Category, string> = {
  malignant: "Malignant",
  precancerous: "Precancerous",
  benign: "Benign",
  "non-lesion": "Non-cancerous condition",
  invalid: "Not skin",
};

export function classByKey(key: string): ClassInfo {
  const found = CLASSES.find((c) => c.key === key);
  if (!found) {
    throw new Error(`Unknown class key: ${key}`);
  }
  return found;
}
