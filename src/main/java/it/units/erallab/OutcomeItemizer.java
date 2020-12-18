package it.units.erallab;

import it.units.erallab.hmsrobots.tasks.locomotion.Footprint;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.malelab.jgea.core.listener.collector.Item;
import it.units.malelab.jgea.core.util.TextPlotter;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * @author eric
 */
public class OutcomeItemizer implements Function<Outcome, List<Item>> {

  private final double startT;
  private final double endT;
  private final double spectrumMinFreq;
  private final double spectrumMaxFreq;
  private final int spectrumSize;

  public OutcomeItemizer(double startT, double endT, double spectrumMinFreq, double spectrumMaxFreq, int spectrumSize) {
    this.startT = startT;
    this.endT = endT;
    this.spectrumMinFreq = spectrumMinFreq;
    this.spectrumMaxFreq = spectrumMaxFreq;
    this.spectrumSize = spectrumSize;
  }

  @Override
  public List<Item> apply(Outcome o) {
    List<Item> items = new ArrayList<>();
    items.addAll(
        List.of(
            new Item("computation.time", o.getComputationTime(), "%4.1f"),
            new Item("time", o.getTime(), "%4.1f"),
            new Item("area.ratio.power", o.getAreaRatioPower(), "%5.1f"),
            new Item("control.power", o.getControlPower(), "%5.1f"),
            new Item("corrected.efficiency", o.getCorrectedEfficiency(), "%6.3f"),
            new Item("distance", o.getDistance(), "%5.1f"),
            new Item("velocity", o.getVelocity(), "%6.3f"),
            new Item(
                "average.posture",
                Grid.toString(o.getAveragePosture(startT, endT), (Predicate<Boolean>) b -> b, "|"),
                "%10.10s"
            )
        ));
    List<Outcome.Mode> xModes = o.getCenterPowerSpectrum(startT, endT, Outcome.Component.X, spectrumMinFreq, spectrumMaxFreq, spectrumSize);
    List<Outcome.Mode> yModes = o.getCenterPowerSpectrum(startT, endT, Outcome.Component.X, spectrumMinFreq, spectrumMaxFreq, spectrumSize);
    items.add(new Item("spectrum.y", TextPlotter.barplot(yModes.stream().mapToDouble(Outcome.Mode::getStrength).toArray(), 8), "%8s"));
    Outcome.Gait g = o.getMainGait(startT, endT);
    items.addAll(List.of(
        new Item("gait.average.touch.area", g == null ? null : g.getAvgTouchArea(), "%5.3f"),
        new Item("gait.coverage", g == null ? null : g.getCoverage(), "%4.2f"),
        new Item("gait.mode.interval", g == null ? null : g.getModeInterval(), "%3.1f"),
        new Item("gait.purity", g == null ? null : g.getPurity(), "%4.2f"),
        new Item("gait.num.unique.footprints", g == null ? null : g.getFootprints().stream().distinct().count(), "%2d"),
        new Item("gait.num.footprints", g == null ? null : g.getFootprints().size(), "%2d"),
        new Item("gait.footprints", g == null ? null : g.getFootprints().stream().map(Footprint::toString).collect(Collectors.joining("|")), "%10.10s")
    ));
    items.addAll(getSpectrum(xModes, Outcome.Component.X));
    items.addAll(getSpectrum(yModes, Outcome.Component.Y));
    return items;
  }

  private List<Item> getSpectrum(List<Outcome.Mode> modes, Outcome.Component component) {
    List<Item> items = new ArrayList<>();
    for (int i = 0; i < spectrumSize; i++) {
      items.add(new Item(
          String.format("spectrum.%s.%d", component.toString().toLowerCase(), i),
          i < modes.size() ? modes.get(i).getStrength() : null,
          "%3.1f")
      );
    }
    return items;
  }
}
