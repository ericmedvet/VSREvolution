package it.units.erallab.builder.misc;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.function.MLP;
import it.units.erallab.builder.function.PruningMLP;
import it.units.erallab.builder.function.RNN;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * @author eric
 */
public class FunctionsGrid implements NamedProvider<PrototypedFunctionBuilder<List<Double>, Grid<TimedRealFunction>>> {

  private final static String INNER_SEPARATOR = "~";

  private final NamedProvider<PrototypedFunctionBuilder<List<Double>, TimedRealFunction>> mapperBuilderProvider =
      NamedProvider.of(Map.ofEntries(
          Map.entry("mlp", new MLP(MultiLayerPerceptron.ActivationFunction.TANH)),
          Map.entry(
              "pMlp",
              new PruningMLP(
                  MultiLayerPerceptron.ActivationFunction.TANH,
                  PruningMultiLayerPerceptron.Context.NETWORK,
                  PruningMultiLayerPerceptron.Criterion.ABS_SIGNAL_MEAN
              )
          ),
          Map.entry("rnn", new RNN())
      ));

  @Override
  public PrototypedFunctionBuilder<List<Double>, Grid<TimedRealFunction>> build(Map<String, String> params) {
    PrototypedFunctionBuilder<List<Double>, TimedRealFunction> itemBuilder = mapperBuilderProvider.build(
        params.get("iB").replace(INNER_SEPARATOR, TOKEN_SEPARATOR).replace(":", PARAM_VALUE_SEPARATOR)
    ).get();
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, Grid<TimedRealFunction>> buildFor(Grid<TimedRealFunction> targetFunctions) {
        return values -> {
          Grid<TimedRealFunction> functions = Grid.create(targetFunctions);
          int c = 0;
          for (Grid.Entry<TimedRealFunction> entry : targetFunctions) {
            if (entry.value() == null) {
              continue;
            }
            int size = itemBuilder.exampleFor(entry.value()).size();
            functions.set(
                entry.key().x(),
                entry.key().y(),
                itemBuilder.buildFor(entry.value()).apply(values.subList(c, c + size))
            );
            c = c + size;
          }
          return functions;
        };
      }

      @Override
      public List<Double> exampleFor(Grid<TimedRealFunction> functions) {
        return Collections.nCopies(
            functions.values().stream()
                .filter(Objects::nonNull)
                .mapToInt(f -> itemBuilder.exampleFor(f).size())
                .sum(),
            0d
        );
      }
    };
  }


}
