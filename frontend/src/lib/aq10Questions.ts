export interface AQ10Question {
  code: string;
  title: string;
  question: string;
}

// AQ-10-Child screening item wording, provided by the project owner.
// Order matches A1_Score..A10_Score in data/raw/autism_child/Autism-Child-Data.arff.
export const AQ10_QUESTIONS: AQ10Question[] = [
  {
    code: "A1",
    title: "Name response",
    question: "Does the child look toward you when you call their name?",
  },
  {
    code: "A2",
    title: "Eye contact",
    question: "How easy is it to establish eye contact with the child?",
  },
  {
    code: "A3",
    title: "Requesting",
    question: "Does the child point toward something they want?",
  },
  {
    code: "A4",
    title: "Sharing interest",
    question: "Does the child point toward something to share their interest with you?",
  },
  {
    code: "A5",
    title: "Pretend play",
    question: "Does the child engage in pretend/imaginative play?",
  },
  {
    code: "A6",
    title: "Following gaze",
    question: "Does the child follow the direction in which another person is looking?",
  },
  {
    code: "A7",
    title: "Comforting",
    question: "When someone is visibly upset, does the child show an attempt to comfort them?",
  },
  {
    code: "A8",
    title: "First words",
    question: "How would you describe the child's first-word development?",
  },
  {
    code: "A9",
    title: "Gestures",
    question: "Does the child use simple gestures, such as waving?",
  },
  {
    code: "A10",
    title: "Unfocused staring",
    question: "Does the child sometimes stare at nothing without an apparent purpose?",
  },
];
