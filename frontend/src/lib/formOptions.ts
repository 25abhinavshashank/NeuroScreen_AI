// Sourced from the training dataset's ARFF attribute definitions
// (data/raw/autism_child/Autism-Child-Data.arff) so form inputs match
// values the model's preprocessing pipeline was actually fit on.

export const GENDER_OPTIONS = [
  { value: "m", label: "Male" },
  { value: "f", label: "Female" },
];

export const YES_NO_OPTIONS = [
  { value: "no", label: "No" },
  { value: "yes", label: "Yes" },
];

export const ETHNICITY_OPTIONS = [
  "White-European",
  "Middle Eastern",
  "Asian",
  "South Asian",
  "Black",
  "Hispanic",
  "Latino",
  "Turkish",
  "Pasifika",
  "Others",
];

export const RELATION_OPTIONS = ["Parent", "Self", "Relative", "Health care professional"];

export const COUNTRY_OPTIONS = [
  "United States",
  "United Kingdom",
  "Jordan",
  "Egypt",
  "Bahrain",
  "Austria",
  "Kuwait",
  "United Arab Emirates",
  "India",
  "Australia",
  "Canada",
  "Saudi Arabia",
  "New Zealand",
  "South Africa",
  "Germany",
  "Netherlands",
  "Other",
];

export const AQ10_ITEM_COUNT = 10;
