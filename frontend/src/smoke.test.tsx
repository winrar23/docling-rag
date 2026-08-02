import { render, screen } from "@testing-library/react";

test("тулчейн жив: рендер + jest-dom", () => {
  render(<p>докель-раг</p>);
  expect(screen.getByText("докель-раг")).toBeInTheDocument();
});
