package it.units.erallab;

import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;

public class CsvQuerier {

  private static final String PREDICATE_DEF = "\\s*=\\s*";
  private static final String PREDICATE_SEP = "\\s*;\\s*";
  private static final String PREDICATE_F_SEP = "\\s*,\\s*";
  private static final String[] PREDICATE_F_PARS = new String[]{"(", ")"};

  private static final String VAR_NAME = "[a-zA-Z][a-zA-Z0-9.]*";

  private static final Map<String, Function<String, String>> FUNCTIONS;

  static {
    FUNCTIONS = Map.of(
        "quantize\\(" + VAR_NAME + ",(?<n>\\d+(\\.\\d+)?)\\)",
        s -> s
    );
  }

  private static class Filter {
    private final String varName;
    private final Function<String, String> processor;
    private final Predicate<String> predicate;

    public Filter(String varName, Function<String, String> processor, Predicate<String> predicate) {
      this.varName = varName;
      this.processor = processor;
      this.predicate = predicate;
    }

    public String getVarName() {
      return varName;
    }

    public Function<String, String> getProcessor() {
      return processor;
    }

    public Predicate<String> getPredicate() {
      return predicate;
    }
  }


  private static Map.Entry<String, Predicate<String>> buildPredicate(String string) {
    String left = string.split(PREDICATE_DEF)[0];
    String right = string.split(PREDICATE_DEF)[1];
    if (!left.contains(PREDICATE_F_PARS[0])) {
      return Map.entry(left, s -> s.equals(right));
    }
    String fName = left.substring(0, left.indexOf(PREDICATE_F_PARS[0]));
    String[] fArgs = left.substring(left.indexOf(PREDICATE_F_PARS[0]) + 1, left.indexOf(PREDICATE_F_PARS[1])).split(PREDICATE_F_SEP);
    if (fName.equals("quant")) {
      double range = Double.parseDouble(fArgs[1]);
      int bins = Integer.parseInt(fArgs[2]);
      return Map.entry(
          fArgs[0],
          s -> Math.floor(Double.parseDouble(s) / range * bins) * range / bins == Double.parseDouble(right)
      );
    }
    return null;
  }

}
